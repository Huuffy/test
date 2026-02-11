"""
Search for Linda across all tables in ERPNextDB
"""

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text, inspect
from db import target_db_url
import pandas as pd

engine = create_engine(target_db_url)
insp = inspect(engine)

print("Searching for 'Linda' across all tables in ERPNextDB...\n")
print("="*80)

# Get all tables
all_tables = insp.get_table_names()
print(f"Total tables to search: {len(all_tables)}\n")

results_summary = []
detailed_results = {}

# Search each table for any column containing 'Linda'
for table_name in all_tables:
    try:
        # Get column info
        columns = insp.get_columns(table_name)
        col_names = [c['name'] for c in columns]

        # Build a WHERE clause that checks all string columns for 'Linda'
        string_cols = []
        for col in columns:
            col_type_str = str(col['type']).upper()
            if any(t in col_type_str for t in ['VARCHAR', 'NVARCHAR', 'TEXT', 'CHAR']):
                string_cols.append(col['name'])

        if not string_cols:
            continue

        # Build LIKE conditions for each string column
        conditions = " OR ".join([f"[{col}] LIKE '%Linda%'" for col in string_cols])

        # Query
        sql = f"SELECT TOP 5 * FROM [{table_name}] WHERE {conditions}"

        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()

            if rows:
                count_sql = f"SELECT COUNT(*) FROM [{table_name}] WHERE {conditions}"
                total_count = conn.execute(text(count_sql)).scalar()

                results_summary.append({
                    'table': table_name,
                    'count': total_count,
                    'string_columns': ', '.join(string_cols[:5])  # First 5 cols
                })

                # Store detailed results
                col_names = list(result.keys())
                detailed_results[table_name] = {
                    'columns': col_names,
                    'rows': [dict(zip(col_names, row)) for row in rows]
                }

                print(f"✓ {table_name}: {total_count} record(s)")

    except Exception as e:
        # Skip tables we can't query
        pass

print("\n" + "="*80)
print(f"\nSUMMARY: Found Linda in {len(results_summary)} table(s)\n")

# Print summary table
if results_summary:
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    # Print detailed results for key tables
    print("\n" + "="*80)
    print("DETAILED RESULTS (first 5 records per table):\n")

    for table_name, data in detailed_results.items():
        print(f"\n### {table_name} ({len(data['rows'])} shown):")
        print("-" * 80)

        for i, row in enumerate(data['rows'], 1):
            print(f"\nRecord {i}:")
            for key, val in row.items():
                if val and 'linda' in str(val).lower():
                    print(f"  {key}: {val}")
else:
    print("No records found containing 'Linda'")
