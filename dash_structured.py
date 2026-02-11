"""
Dash with Structured Output - Gets both SQL and natural language response
Uses Pydantic response_model to force structured output
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import json
from typing import Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.sql import SQLTools
from db import target_db_url, get_agent_db
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

def generate_sql_query(question: str) -> SqlResponse:
    """Generate structured SQL query for the question"""
    print(f"\nGenerating SQL for: {question}")
    print("-" * 80)

    prompt = f"""Generate a SQL Server query to find information about: {question}

Search relevant tables (J_CustomerContact, CustomerInquiryMaster, J_Job, etc.)
Return TOP 10 results maximum.
Use LIKE for name searches."""

    try:
        # Try structured output
        response = sql_agent.run(prompt, response_model=SqlResponse)

        if hasattr(response, 'content') and isinstance(response.content, SqlResponse):
            return response.content
        elif isinstance(response, SqlResponse):
            return response
        else:
            # Fallback: parse from text
            print("WARNING: Structured output failed, using fallback...")
            raise ValueError("Need fallback")

    except Exception as e:
        print(f"WARNING: Structured output failed: {e}")
        print("Using fallback SQL generation...")

        # Fallback: simple search
        search_term = question.split()[-1] if ' ' in question else question

        return SqlResponse(
            thinking=f"Searching for '{search_term}' in customer contact tables",
            tables_to_query=["J_CustomerContact"],
            sql_query=f"""SELECT TOP 10
    CustomerContactId,
    CustomerId,
    FirstName,
    LastName,
    EmailAddress,
    PhoneNumber,
    MobilePhone,
    Position,
    Title
FROM J_CustomerContact
WHERE LastName LIKE '%{search_term}%'
   OR FirstName LIKE '%{search_term}%'
   OR EmailAddress LIKE '%{search_term}%'""",
            explanation=f"Search J_CustomerContact for any records matching '{search_term}'"
        )

def execute_sql(sql_query: str) -> list[dict]:
    """Execute SQL and return results"""
    print(f"\nExecuting SQL...")
    print("-" * 80)
    print(sql_query)
    print("-" * 80)

    sql_tool = SQLTools(db_url=target_db_url)

    try:
        result_json = sql_tool.run_sql_query(sql_query, limit=None)
        results = json.loads(result_json)
        print(f"\nFound {len(results)} record(s)")
        return results
    except Exception as e:
        print(f"ERROR: SQL Error: {e}")
        return []

def introduce_person(question: str, results: list[dict], sql_response: SqlResponse) -> PersonIntroduction:
    """Generate natural language introduction based on database results"""
    print(f"\nGenerating introduction...")
    print("-" * 80)

    if not results:
        return PersonIntroduction(
            name="Unknown",
            role_title="Not found",
            company="N/A",
            contact_info="No records found",
            relationship="Unknown",
            summary="No records found in the database for this person.",
            data_sources=[]
        )

    # Use Ollama directly for better natural language
    context = f"""Based on this database query result, introduce this person professionally:

Question: {question}

Database Record:
{json.dumps(results[0], indent=2, default=str)}

Additional context:
- Found in table: {sql_response.tables_to_query[0] if sql_response.tables_to_query else 'Unknown'}
- Total records found: {len(results)}

Create a professional 2-3 sentence introduction that explains:
1. Who they are (name, title)
2. Their role/position
3. How they relate to the company (customer contact, vendor, etc)
4. Key contact information

Be conversational and informative."""

    response = ollama.chat(
        model='qwen2.5:3b',
        messages=[{
            'role': 'user',
            'content': context
        }]
    )

    intro_text = response['message']['content']

    # Extract structured data from first result
    first_record = results[0]

    name = f"{first_record.get('FirstName', '')} {first_record.get('LastName', '')}".strip()
    if not name:
        name = first_record.get('Contact', first_record.get('Attention', 'Unknown'))

    return PersonIntroduction(
        name=name,
        role_title=first_record.get('Position', first_record.get('Title', 'Not specified')),
        company=first_record.get('CustomerName', 'Associated customer'),
        contact_info=f"Email: {first_record.get('EmailAddress', first_record.get('EMail_Address', 'N/A'))}, Phone: {first_record.get('PhoneNumber', first_record.get('Phone', 'N/A'))}",
        relationship="Customer Contact" if 'CustomerContact' in sql_response.tables_to_query[0] else "Contact",
        summary=intro_text,
        data_sources=sql_response.tables_to_query
    )

def query_and_introduce(question: str):
    """Complete flow: Question -> SQL -> Results -> Introduction"""
    print("\n" + "=" * 80)
    print(f"DASH QUERY: {question}")
    print("=" * 80)

    # Step 1: Generate SQL
    sql_response = generate_sql_query(question)

    print(f"\nAnalysis:")
    print(f"  Thinking: {sql_response.thinking}")
    print(f"  Tables: {', '.join(sql_response.tables_to_query)}")
    print(f"  Explanation: {sql_response.explanation}")

    # Step 2: Execute SQL
    results = execute_sql(sql_response.sql_query)

    if not results:
        print("\nNo results found.")
        return None, None, None

    # Step 3: Show raw results
    print(f"\nDatabase Records:")
    print("=" * 80)
    for i, record in enumerate(results[:3], 1):
        print(f"\nRecord {i}:")
        for key, val in record.items():
            if val:
                print(f"  {key}: {val}")

    if len(results) > 3:
        print(f"\n  ... and {len(results) - 3} more record(s)")

    # Step 4: Generate introduction
    introduction = introduce_person(question, results, sql_response)

    print(f"\nINTRODUCTION:")
    print("=" * 80)
    print(f"Name: {introduction.name}")
    print(f"Role: {introduction.role_title}")
    print(f"Contact: {introduction.contact_info}")
    print(f"\n{introduction.summary}")
    print(f"\nData from: {', '.join(introduction.data_sources)}")

    return sql_response, results, introduction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Who would you like to know about? ")
    else:
        question = " ".join(sys.argv[1:])

    query_and_introduce(question)
