"""
Dash with Multi-Table Search - Searches across ALL relevant tables
Usage: python dash_multitable.py "Linda"
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import json
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.sql import SQLTools
from db import target_db_url, get_agent_db
from sqlalchemy import create_engine, text, inspect
import ollama

# Define response schema
class SqlResponse(BaseModel):
    """Structured SQL response from Dash"""
    thinking: str = Field(description="Your reasoning about what tables to check and why")
    tables_to_query: list[str] = Field(description="List of table names you'll query")
    sql_query: str = Field(description="The complete T-SQL query to execute")
    explanation: str = Field(description="What this query does and why")

class PersonIntroduction(BaseModel):
    """Introduction/summary of a person based on database records"""
    name: str = Field(description="Full name of the person")
    role_title: str = Field(description="Their job title or position")
    company: str = Field(description="Company they're associated with")
    contact_info: str = Field(description="Email and/or phone number")
    relationship: str = Field(description="Their relationship to the organization (customer, vendor, etc)")
    summary: str = Field(description="2-3 sentence introduction describing who they are and key details")
    data_sources: list[str] = Field(description="Which tables the information came from")

# Simple agent for SQL generation
sql_agent = Agent(
    name="SQL Generator",
    model=Ollama(id="qwen2.5:3b"),
    instructions="""You are a SQL expert for ERPNextDB (SQL Server with 1406 tables).

Key tables for people:
- J_CustomerContact: Main contacts (FirstName, LastName, EmailAddress, PhoneNumber, Position)
- CustomerInquiryMaster: Inquiries (Contact, EMail_Address, Comment, CustomerNumber)
- J_Job: Jobs/Orders (Attention, Description, JobNumber)
- J_Customer: Customers (CustomerNumber, Name, AddressLine1, City)

SQL Server rules:
- Use TOP N (not LIMIT)
- Use [brackets] for table/column names with spaces
- Use LIKE '%text%' for searches
- Can use UNION ALL to combine results from multiple tables""",
    markdown=False,
)

def search_all_tables(search_term: str, max_tables: int = 10) -> List[Dict]:
    """
    Search across ALL database tables for the search term
    Returns list of {table, count, records}
    """
    print(f"\nSearching across all tables for: {search_term}")
    print("-" * 80)

    search_terms = search_term.split()
    engine = create_engine(target_db_url)
    insp = inspect(engine)
    all_tables = insp.get_table_names()

    found_results = []

    for table_name in all_tables:
        if len(found_results) >= max_tables:
            break

        try:
            columns = insp.get_columns(table_name)
            string_cols = []
            for col in columns:
                col_type_str = str(col['type']).upper()
                if any(t in col_type_str for t in ['VARCHAR', 'NVARCHAR', 'TEXT', 'CHAR']):
                    string_cols.append(col['name'])

            if not string_cols:
                continue

            # Build conditions for each search term
            term_conditions = []
            for term in search_terms:
                col_conditions = [f"[{col}] LIKE '%{term}%'" for col in string_cols]
                term_conditions.append(f"({' OR '.join(col_conditions)})")

            # All terms must be found (AND logic)
            where_clause = ' AND '.join(term_conditions)

            sql = f"SELECT TOP 10 * FROM [{table_name}] WHERE {where_clause}"

            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()

                if rows:
                    count_sql = f"SELECT COUNT(*) FROM [{table_name}] WHERE {where_clause}"
                    total_count = conn.execute(text(count_sql)).scalar()

                    col_names = list(result.keys())
                    records = [dict(zip(col_names, row)) for row in rows]

                    found_results.append({
                        'table': table_name,
                        'count': total_count,
                        'columns': col_names,
                        'records': records
                    })

                    print(f"  Found in {table_name}: {total_count} record(s)")
        except Exception as e:
            pass

    print(f"\nFound results in {len(found_results)} table(s)")
    return found_results

def combine_results(table_results: List[Dict]) -> tuple[list[dict], list[str]]:
    """Combine results from multiple tables into single list"""
    all_records = []
    all_tables = []

    for result in table_results:
        table_name = result['table']
        all_tables.append(table_name)

        for record in result['records']:
            # Add table source to each record
            record_with_source = record.copy()
            record_with_source['_source_table'] = table_name
            all_records.append(record_with_source)

    return all_records, all_tables

def introduce_person(question: str, table_results: List[Dict]) -> PersonIntroduction:
    """Generate natural language introduction based on multi-table results"""
    print(f"\nGenerating introduction from {len(table_results)} table(s)...")
    print("-" * 80)

    if not table_results:
        return PersonIntroduction(
            name="Unknown",
            role_title="Not found",
            company="N/A",
            contact_info="No records found",
            relationship="Unknown",
            summary="No records found in the database for this person.",
            data_sources=[]
        )

    # Combine all records
    all_records, table_names = combine_results(table_results)

    # Use Ollama to create comprehensive introduction
    context = f"""Based on database searches across {len(table_results)} tables, introduce this person professionally:

Question: {question}

Found in tables: {', '.join(table_names)}

Sample records:
{json.dumps(all_records[:5], indent=2, default=str)}

Total records: {sum(r['count'] for r in table_results)}

Create a comprehensive 2-3 sentence introduction that explains:
1. Who they are (name, title from most detailed record)
2. Their role/position
3. How they relate to the company (customer contact, vendor, etc)
4. Key contact information (email, phone from any table)
5. Any other relevant details found across tables

Be conversational and informative. Mention that information was found across multiple systems/tables."""

    response = ollama.chat(
        model='qwen2.5:3b',
        messages=[{
            'role': 'user',
            'content': context
        }]
    )

    intro_text = response['message']['content']

    # Extract structured data from first result's first record
    first_record = table_results[0]['records'][0]

    # Try to get name from various fields
    name = ""
    if 'FirstName' in first_record and 'LastName' in first_record:
        name = f"{first_record.get('FirstName', '')} {first_record.get('LastName', '')}".strip()
    elif 'Contact' in first_record:
        name = first_record.get('Contact', '')
    elif 'Attention' in first_record:
        name = first_record.get('Attention', '')
    elif 'Name' in first_record:
        name = first_record.get('Name', '')

    if not name:
        name = "Unknown"

    # Get contact info from all records
    emails = []
    phones = []
    for result in table_results:
        for record in result['records']:
            if email := record.get('EmailAddress') or record.get('EMail_Address') or record.get('Email'):
                if email and email not in emails:
                    emails.append(str(email))
            if phone := record.get('PhoneNumber') or record.get('Phone') or record.get('MobilePhone'):
                if phone and phone not in phones:
                    phones.append(str(phone))

    contact_info = f"Email: {', '.join(emails[:2]) if emails else 'N/A'}, Phone: {', '.join(phones[:2]) if phones else 'N/A'}"

    return PersonIntroduction(
        name=name,
        role_title=first_record.get('Position', first_record.get('Title', 'Not specified')),
        company=first_record.get('CustomerName', 'Associated organization'),
        contact_info=contact_info,
        relationship=f"Found in {len(table_results)} system(s)",
        summary=intro_text,
        data_sources=table_names
    )

def query_and_introduce(question: str):
    """Complete flow: Question -> Multi-table Search -> Results -> Introduction"""
    print("\n" + "=" * 80)
    print(f"DASH MULTI-TABLE QUERY: {question}")
    print("=" * 80)

    # Search across all tables
    search_term = question.split()[-1] if ' ' in question else question
    table_results = search_all_tables(search_term, max_tables=10)

    if not table_results:
        print("\nNo results found in any table.")
        return None, None, None

    # Combine all results
    all_records, table_names = combine_results(table_results)

    # Show summary
    print(f"\nDatabase Records Summary:")
    print("=" * 80)
    print(f"Total records found: {len(all_records)} across {len(table_names)} table(s)")
    print(f"Tables: {', '.join(table_names[:5])}")
    if len(table_names) > 5:
        print(f"        ... and {len(table_names) - 5} more")

    # Show sample records
    print(f"\nSample Records:")
    for i, record in enumerate(all_records[:3], 1):
        print(f"\nRecord {i} (from {record.get('_source_table', 'Unknown')}):")
        for key, val in record.items():
            if val and key != '_source_table':
                print(f"  {key}: {val}")

    if len(all_records) > 3:
        print(f"\n  ... and {len(all_records) - 3} more record(s)")

    # Generate introduction
    introduction = introduce_person(question, table_results)

    print(f"\nINTRODUCTION:")
    print("=" * 80)
    print(f"Name: {introduction.name}")
    print(f"Role: {introduction.role_title}")
    print(f"Contact: {introduction.contact_info}")
    print(f"\n{introduction.summary}")
    print(f"\nData from {len(introduction.data_sources)} table(s): {', '.join(introduction.data_sources)}")

    # Create a fake SqlResponse for compatibility with visualizer
    sql_response = SqlResponse(
        thinking=f"Searching across all database tables for '{search_term}'",
        tables_to_query=table_names,
        sql_query=f"-- Multi-table search across {len(table_names)} tables\n-- Search term: {search_term}\n-- Found {len(all_records)} total records",
        explanation=f"Searched {len(table_names)} tables and found {len(all_records)} matching records"
    )

    return sql_response, all_records, introduction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Who would you like to know about? ")
    else:
        question = " ".join(sys.argv[1:])

    query_and_introduce(question)
