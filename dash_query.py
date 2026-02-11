"""
Dash Query with Visualization - Uses Dash agent for SQL + separate LLM for introduction

Usage: python dash_query.py "who is Art Hoelke"

Flow:
1. Dash agent (qwen2.5:3b) → introspects database → generates SQL
2. Execute SQL → get results
3. Ollama (qwen2.5:3b) → generates natural language introduction
4. Visualizer → shows SQL + Introduction + Results
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import json
import re
from typing import Optional
from pydantic import BaseModel, Field
import ollama
from dash.agents import dash
from agno.tools.sql import SQLTools
from db import target_db_url

class SqlResponse(BaseModel):
    """SQL response extracted from Dash"""
    thinking: str = Field(description="Reasoning about the query")
    tables_to_query: list[str] = Field(description="Tables being queried")
    sql_query: str = Field(description="The SQL query")
    explanation: str = Field(description="What the query does")

class PersonIntroduction(BaseModel):
    """Natural language introduction"""
    name: str
    role_title: str
    company: str
    contact_info: str
    relationship: str
    summary: str
    data_sources: list[str]

def extract_sql_from_response(response_text: str) -> Optional[str]:
    """Extract SQL query from Dash's markdown response"""
    # Look for SQL code blocks
    sql_pattern = r'```sql\n(.*?)\n```'
    matches = re.findall(sql_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()

    # Look for SELECT statements
    select_pattern = r'(SELECT\s+.*?(?:;|$))'
    matches = re.findall(select_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip().rstrip(';')

    return None

def query_with_dash(question: str) -> tuple[Optional[str], Optional[str]]:
    """Use Dash agent to generate SQL"""
    print(f"\n{'='*80}")
    print(f"DASH AGENT: Analyzing question")
    print(f"{'='*80}")
    print(f"Question: {question}\n")

    # Direct prompt that forces SQL generation
    prompt = f"""Generate a SQL Server query to answer: {question}

INSTRUCTIONS:
- Search the J_CustomerContact table
- Use LIKE for fuzzy matching on FirstName, LastName, EmailAddress
- Return TOP 10 results
- Use T-SQL syntax (TOP not LIMIT, [brackets] for names)

OUTPUT REQUIRED:
Provide your SQL query in this exact format:
```sql
SELECT TOP 10 ...
FROM ...
WHERE ...
```"""

    print("Calling Dash agent (this will introspect the database)...")
    print("-" * 80)

    # Run Dash agent
    run_response = dash.run(prompt, stream=False)

    response_text = run_response.content if hasattr(run_response, 'content') else str(run_response)

    print("\nDash Response:")
    print("-" * 80)
    print(response_text)
    print("-" * 80)

    # Extract SQL
    sql = extract_sql_from_response(response_text)

    if sql:
        print(f"\n[OK] Extracted SQL query")
        return sql, response_text
    else:
        print("\n[ERROR] Could not extract SQL from response")
        return None, response_text

def execute_sql(sql_query: str) -> list[dict]:
    """Execute SQL and return results"""
    print(f"\n{'='*80}")
    print(f"EXECUTING SQL")
    print(f"{'='*80}")
    print(sql_query)
    print("-" * 80)

    sql_tool = SQLTools(db_url=target_db_url)

    try:
        result_json = sql_tool.run_sql_query(sql_query, limit=None)
        results = json.loads(result_json)
        print(f"\n[OK] Found {len(results)} record(s)")
        return results
    except Exception as e:
        print(f"\n[ERROR] SQL Error: {e}")
        return []

def generate_introduction(question: str, results: list[dict], sql: str) -> PersonIntroduction:
    """Generate natural language introduction using Ollama"""
    print(f"\n{'='*80}")
    print(f"GENERATING INTRODUCTION")
    print(f"{'='*80}")

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

    # Prepare context for LLM
    context = f"""Based on this database query result, introduce this person professionally:

Question: {question}

SQL Query Used:
{sql}

Database Records Found: {len(results)}

First Record:
{json.dumps(results[0], indent=2, default=str)}

Additional Records: {len(results) - 1 if len(results) > 1 else 0}

Create a professional 2-3 sentence introduction that explains:
1. Who they are (name, title from the record)
2. Their role/position
3. How they relate to the organization (customer contact, vendor, employee, etc)
4. Key contact information (email, phone if available)

Be conversational and informative. Only use information that's actually in the database records."""

    print("Calling Ollama for natural language generation...")

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

    # Try to get name
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
        # Try to extract from any field with "name" in it
        for key, val in first_record.items():
            if 'name' in key.lower() and val:
                name = str(val)
                break

    name = name if name else "Unknown"

    # Get contact info
    email = (first_record.get('EmailAddress') or
             first_record.get('EMail_Address') or
             first_record.get('Email') or
             'N/A')

    phone = (first_record.get('PhoneNumber') or
             first_record.get('Phone') or
             first_record.get('MobilePhone') or
             'N/A')

    contact_info = f"Email: {email}, Phone: {phone}"

    # Guess table name from SQL
    table_match = re.search(r'FROM\s+\[?(\w+)\]?', sql, re.IGNORECASE)
    table_name = table_match.group(1) if table_match else "Unknown"

    print(f"\n[OK] Introduction generated")

    return PersonIntroduction(
        name=name,
        role_title=first_record.get('Position', first_record.get('Title', 'Not specified')),
        company=first_record.get('CustomerName', first_record.get('CompanyName', 'Associated organization')),
        contact_info=contact_info,
        relationship=f"Found in {table_name}",
        summary=intro_text,
        data_sources=[table_name]
    )

def show_visualizer(question: str, sql: str, results: list[dict], introduction: PersonIntroduction, dash_response: str):
    """Show results in GUI visualizer"""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from datetime import datetime
    import pandas as pd

    def export_to_excel():
        if not results:
            messagebox.showwarning("No Data", "No results to export.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile=f"{introduction.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        )
        if not path:
            return

        df = pd.DataFrame(results)
        df.to_excel(path, index=False, engine="openpyxl")
        messagebox.showinfo("Exported", f"Results saved to:\n{path}")

    root = tk.Tk()
    root.title(f"Dash Query — {question[:50]}")
    root.geometry("1200x900")
    root.configure(bg="#1e1e2e")

    # Styling
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", background="#2b2b3d", foreground="white",
                    fieldbackground="#2b2b3d", rowheight=28, font=("Consolas", 10))
    style.configure("Treeview.Heading", background="#3b3b5b", foreground="white",
                    font=("Consolas", 10, "bold"))
    style.map("Treeview", background=[("selected", "#4b4b7b")])

    # Question
    q_frame = tk.Frame(root, bg="#1e1e2e")
    q_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
    tk.Label(q_frame, text="Question:", bg="#1e1e2e", fg="#89b4fa",
             font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
    tk.Label(q_frame, text=question, bg="#1e1e2e", fg="white",
             font=("Consolas", 11), wraplength=1000).pack(side=tk.LEFT, padx=5)

    # Introduction
    intro_frame = tk.LabelFrame(root, text="Dash's Introduction", bg="#1e1e2e", fg="#f9e2af",
                                font=("Consolas", 10, "bold"), bd=2)
    intro_frame.pack(fill=tk.X, padx=10, pady=5)

    intro_text = tk.Text(intro_frame, height=6, bg="#2b2b3d", fg="#cdd6f4",
                        font=("Consolas", 10), wrap=tk.WORD, relief=tk.FLAT)

    intro_content = f"""Name: {introduction.name}
Role: {introduction.role_title}
Contact: {introduction.contact_info}

{introduction.summary}

Data source: {', '.join(introduction.data_sources)}"""

    intro_text.insert(tk.END, intro_content)
    intro_text.configure(state=tk.DISABLED)
    intro_text.pack(fill=tk.X, padx=5, pady=5)

    # Dash's reasoning
    reasoning_frame = tk.LabelFrame(root, text="Dash's Analysis", bg="#1e1e2e", fg="#f5c2e7",
                                   font=("Consolas", 10, "bold"))
    reasoning_frame.pack(fill=tk.X, padx=10, pady=5)

    reasoning_text = tk.Text(reasoning_frame, height=8, bg="#2b2b3d", fg="#cdd6f4",
                            font=("Consolas", 9), wrap=tk.WORD, relief=tk.FLAT)
    reasoning_text.insert(tk.END, dash_response)
    reasoning_text.configure(state=tk.DISABLED)
    reasoning_text.pack(fill=tk.X, padx=5, pady=5)

    # SQL Query
    sql_frame = tk.LabelFrame(root, text="Generated SQL Query", bg="#1e1e2e", fg="#a6e3a1",
                              font=("Consolas", 10, "bold"))
    sql_frame.pack(fill=tk.X, padx=10, pady=5)

    sql_text = tk.Text(sql_frame, height=6, bg="#2b2b3d", fg="#cdd6f4",
                      font=("Consolas", 9), wrap=tk.WORD, relief=tk.FLAT)
    sql_text.insert(tk.END, sql)
    sql_text.configure(state=tk.DISABLED)
    sql_text.pack(fill=tk.X, padx=5, pady=5)

    # Results table
    table_frame = tk.LabelFrame(root, text=f"Database Records ({len(results)} rows)",
                                bg="#1e1e2e", fg="#f9e2af", font=("Consolas", 10, "bold"))
    table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    if results:
        columns = list(results[0].keys())
        tree = ttk.Treeview(table_frame, columns=columns, show="headings")

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=max(120, len(col) * 10), minwidth=100)

        for row in results:
            values = [str(row.get(c, "")) for c in columns]
            tree.insert("", tk.END, values=values)

        v_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    else:
        no_data_label = tk.Label(table_frame, text="No results found",
                                bg="#2b2b3d", fg="#cdd6f4", font=("Consolas", 12))
        no_data_label.pack(expand=True)

    # Buttons
    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

    if results:
        export_btn = tk.Button(
            btn_frame, text="Export to Excel", bg="#a6e3a1", fg="#1e1e2e",
            font=("Consolas", 10, "bold"), relief=tk.FLAT, padx=15, pady=5,
            command=export_to_excel
        )
        export_btn.pack(side=tk.RIGHT, padx=5)

    close_btn = tk.Button(
        btn_frame, text="Close", bg="#f38ba8", fg="#1e1e2e",
        font=("Consolas", 10, "bold"), relief=tk.FLAT, padx=15, pady=5,
        command=root.destroy
    )
    close_btn.pack(side=tk.RIGHT, padx=5)

    root.mainloop()

def main():
    if len(sys.argv) < 2:
        print("Usage: python dash_query.py \"your question\"")
        print('Example: python dash_query.py "who is Art Hoelke"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    # Step 1: Use Dash agent to generate SQL
    sql, dash_response = query_with_dash(question)

    if not sql:
        print("\nERROR: Dash could not generate SQL query")
        print("Please try rephrasing your question or check the database connection")
        sys.exit(1)

    # Step 2: Execute SQL
    results = execute_sql(sql)

    if not results:
        print("\nNo results found. Try a different search term.")
        # Still show visualizer with empty results

    # Step 3: Generate introduction
    introduction = generate_introduction(question, results, sql)

    # Step 4: Show in visualizer
    print(f"\n{'='*80}")
    print("OPENING VISUALIZER")
    print(f"{'='*80}\n")

    show_visualizer(question, sql, results, introduction, dash_response)

if __name__ == "__main__":
    main()
