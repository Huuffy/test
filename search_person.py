"""
Search for any person across all database tables
Usage: python search_person.py "Mark Elinski"
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text, inspect
from db import target_db_url
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python search_person.py \"Name to search\"")
    sys.exit(1)

search_name = sys.argv[1]
search_terms = search_name.split()

engine = create_engine(target_db_url)
insp = inspect(engine)

print(f"Searching for: {search_name}")
print("="*80)

all_tables = insp.get_table_names()
found_results = []

for table_name in all_tables:
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

                print(f"Found in {table_name}: {total_count} record(s)")
    except Exception as e:
        pass

print("\n" + "="*80)

if found_results:
    print(f"\nFOUND IN {len(found_results)} TABLE(S)\n")

    for result in found_results:
        print(f"\n{'='*80}")
        print(f"TABLE: {result['table']} ({result['count']} total record(s))")
        print(f"{'='*80}")

        for i, record in enumerate(result['records'][:3], 1):
            print(f"\nRecord {i}:")
            for key, val in record.items():
                if val and any(term.lower() in str(val).lower() for term in search_terms):
                    print(f"  {key}: {val}")
else:
    print(f"\nNO RECORDS FOUND for '{search_name}'")
    print("\nTry:")
    print("  - Different spelling")
    print("  - Just last name")
    print("  - Just first name")
