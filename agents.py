"""
Dash Agents
===========

Text-to-SQL agent using Ollama (local LLM) + LanceDB + SQLite.
Target database: SQL Server via pyodbc (Windows Auth).

Test: python -m dash.agents
"""

from os import getenv
from pathlib import Path

from agno.agent import Agent
from dash.schemas import DashSqlResponse
from agno.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.learn import (
    LearnedKnowledgeConfig,
    LearningMachine,
    LearningMode,
    UserMemoryConfig,
    UserProfileConfig,
)
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.sql import SQLTools
from agno.vectordb.lancedb import LanceDb, SearchType

from dash.context.business_rules import BUSINESS_CONTEXT
from dash.context.semantic_model import SEMANTIC_MODEL_STR
from dash.tools import create_introspect_schema_tool, create_save_validated_query_tool
from db import db_url, get_agent_db

# ============================================================================
# Configuration
# ============================================================================

OLLAMA_MODEL = getenv("OLLAMA_MODEL", "mistral")
OLLAMA_EMBED_MODEL = getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_DIMENSIONS = int(getenv("OLLAMA_EMBED_DIMENSIONS", "768"))

# Azure OpenAI Configuration
USE_AZURE_OPENAI = bool(getenv("AZURE_OPENAI_ENDPOINT"))  # Auto-detect Azure config
AZURE_OPENAI_ENDPOINT = getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_MODEL = getenv("AZURE_OPENAI_MODEL", "gpt-4o")

# Model selection priority: Azure OpenAI > Regular OpenAI > Ollama
USE_OPENAI = bool(getenv("OPENAI_API_KEY")) and not USE_AZURE_OPENAI

# Local data directory for LanceDB and SQLite
DATA_DIR = Path(getenv("DATA_DIR", Path(__file__).parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

LANCEDB_URI = str(DATA_DIR / "lancedb")

# ============================================================================
# Database & Knowledge
# ============================================================================

agent_db = get_agent_db()

# Embedder: Ollama (local, free)
embedder = OllamaEmbedder(id=OLLAMA_EMBED_MODEL, dimensions=OLLAMA_EMBED_DIMENSIONS)

# KNOWLEDGE: Static, curated (table schemas, validated queries, business rules)
dash_knowledge = Knowledge(
    name="Dash Knowledge",
    vector_db=LanceDb(
        uri=LANCEDB_URI,
        table_name="dash_knowledge",
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

# LEARNINGS: Dynamic, discovered (error patterns, gotchas, user corrections)
dash_learnings = Knowledge(
    name="Dash Learnings",
    vector_db=LanceDb(
        uri=LANCEDB_URI,
        table_name="dash_learnings",
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

# ============================================================================
# Tools
# ============================================================================

save_validated_query = create_save_validated_query_tool(dash_knowledge)
introspect_schema = create_introspect_schema_tool(db_url)

base_tools: list = [
    SQLTools(db_url=db_url),
    save_validated_query,
    introspect_schema,
]

# ============================================================================
# Instructions
# ============================================================================

INSTRUCTIONS = f"""\
You are Dash, a self-learning data agent that provides **insights**, not just query results.
You are connected to a **Microsoft SQL Server** database.

## Your Purpose

You are the user's data analyst — one that never forgets, never repeats mistakes,
and gets smarter with every query.

You don't just fetch data. You interpret it, contextualize it, and explain what it means.
You remember the gotchas, the type mismatches, the date formats that tripped you up before.

Your goal: make the user look like they've been working with this data for years.

## YOUR CORE BEHAVIOR - NON-INTERACTIVE MODE

**CRITICAL**: You MUST generate SQL for EVERY question. NEVER ask for clarification.

❌ FORBIDDEN - DO NOT say:
- "Would you like me to...?"
- "I can help you find..."
- "Should I search for...?"
- "Could you clarify...?"
- "Let me know if..."
- "I am unable to finish..."

✅ REQUIRED - YOU MUST:
- Generate SQL query immediately after using discovery tools
- Use `run_sql_query` to execute the query
- Return results without asking permission
- If uncertain, make your best guess and generate SQL anyway
- Wrap SQL in ```sql code blocks for easy extraction

## Two Knowledge Systems

**Knowledge** (static, curated):
- Table schemas, validated queries, business rules
- Searched automatically before each response
- Add successful queries here with `save_validated_query`

**Learnings** (dynamic, discovered):
- Patterns YOU discover through errors and fixes
- Type gotchas, date formats, column quirks
- Search with `search_learnings`, save with `save_learning`

## Workflow - NON-INTERACTIVE MODE

1. Search context: Use `search_knowledge_base` and `search_learnings` for table info
2. Generate SQL immediately: Write T-SQL query (no asking permission)
3. **Use JOINs for multi-table queries**: Connect related tables via foreign keys
4. Execute automatically: Call `run_sql_query` with your generated SQL
5. Save learnings: If errors occur, use `introspect_schema` → fix → `save_learning`

**T-SQL / SQL Server syntax**:
   - Use `TOP 50` instead of `LIMIT 50`
   - Use `[brackets]` for column/table names with spaces or reserved words
   - Use `CAST()` or `CONVERT()` for type conversions
   - Use `ISNULL()` instead of `COALESCE()` where appropriate
   - Use `GETDATE()` instead of `NOW()`

## When to save_learning

After fixing a type error:
```
save_learning(
  title="column_name is NVARCHAR not INT",
  learning="Use string comparison for this column"
)
```

After discovering a date format:
```
save_learning(
  title="date column parsing",
  learning="Use CONVERT(date, column, style) for date extraction"
)
```

After a user corrects you:
```
save_learning(
  title="Business rule about table X",
  learning="Details about the correction"
)
```

## Multi-Table Queries (JOINS)

When questions require data from multiple tables, ALWAYS use JOINs:

```sql
-- Example: Customer contacts with company info
SELECT TOP 50
    cc.FirstName,
    cc.LastName,
    cc.EmailAddress,
    c.CustomerName,
    c.City
FROM J_CustomerContact cc
INNER JOIN J_Customer c ON cc.CustomerId = c.CustomerId
WHERE cc.LastName LIKE '%search%'
```

Common join patterns:
- J_CustomerContact.CustomerId = J_Customer.CustomerId
- J_Job.CustomerId = J_Customer.CustomerId
- Always use table aliases for clarity

## Insights, Not Just Data

| Bad | Good |
|-----|------|
| "3 results found" | "Found 3 matching contacts — Linda appears in both Sales and Support departments" |
| "Query returned 50 rows" | "50 orders this month, 15% above the monthly average — mostly from Region A" |

## SQL Server Rules

- Use SELECT TOP 50 by default (not LIMIT)
- Never SELECT * — specify columns
- ORDER BY for top-N queries
- No DROP, DELETE, UPDATE, INSERT
- Use [brackets] for identifiers when needed
- String comparisons: use LIKE for partial matches

---

## SEMANTIC MODEL

{SEMANTIC_MODEL_STR}
---

{BUSINESS_CONTEXT}\
"""

# ============================================================================
# Create Agent
# ============================================================================

# Select model: Azure OpenAI > OpenAI > Ollama
if USE_AZURE_OPENAI:
    # Azure OpenAI requires api-version as query parameter
    model = OpenAIChat(
        id=AZURE_OPENAI_MODEL,
        api_key=AZURE_OPENAI_API_KEY,
        base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_MODEL}",
        extra_headers={"api-key": AZURE_OPENAI_API_KEY},
        extra_query={"api-version": AZURE_OPENAI_API_VERSION}
    )
    print(f"[CONFIG] Using Azure OpenAI: {AZURE_OPENAI_MODEL} at {AZURE_OPENAI_ENDPOINT}")
elif USE_OPENAI:
    model = OpenAIChat(id="gpt-3.5-turbo")
    print(f"[CONFIG] Using OpenAI: gpt-3.5-turbo")
else:
    model = Ollama(id=OLLAMA_MODEL)
    print(f"[CONFIG] Using Ollama: {OLLAMA_MODEL}")

dash = Agent(
    name="Dash",
    model=model,
    db=agent_db,
    instructions=INSTRUCTIONS,
    # Knowledge (static)
    knowledge=dash_knowledge,
    search_knowledge=True,
    # Learning (provides search_learnings, save_learning, user profile, user memory)
    learning=LearningMachine(
        knowledge=dash_learnings,
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
    tools=base_tools,
    # Context
    add_datetime_to_context=True,
    add_history_to_context=True,
    read_chat_history=True,
    num_history_runs=5,
    markdown=True,
)

# Reasoning variant - adds multi-step reasoning capabilities
reasoning_dash = dash.deep_copy(
    update={
        "name": "Reasoning Dash",
        "tools": base_tools + [ReasoningTools(add_instructions=True)],
    }
)

# ============================================================================
# Non-Interactive SQL Generation Agent
# ============================================================================

# Non-interactive variant - allows tool usage but no questions
dash_sql = dash.deep_copy(
    update={
        "name": "Dash SQL Generator",
        "description": "Non-interactive text-to-SQL agent that generates queries immediately using discovery tools. MUST always generate SQL.",
        "markdown": False,
        "add_history_to_context": False,
        "tool_call_limit": 20,  # Increased to allow more discovery
    }
)

# ============================================================================
# Non-Interactive Convenience Function
# ============================================================================

def query_sql_noninteractive(
    question: str,
    execute: bool = True,
    return_insights: bool = True
) -> dict:
    """
    Non-interactive SQL generation: question → SQL → results

    Now allows tool usage for discovery while maintaining non-interactive behavior.

    Args:
        question: Natural language question
        execute: If True, execute SQL and return results
        return_insights: If True, generate natural language insights

    Returns:
        dict with: sql, reasoning, results (if execute), insights (if return_insights), tools_used
    """
    from agno.tools.sql import SQLTools
    import json
    import re

    print(f"[1/3] Generating SQL with tool usage for: {question}")

    # Run agent - it will use tools for discovery
    response = dash_sql.run(question, stream=False)

    # Extract content
    content = response.content if hasattr(response, 'content') else str(response)

    # Track which tools were used and extract SQL from tool calls
    tools_used = []
    sql_query = None

    if hasattr(response, 'messages'):
        for msg in response.messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for call in msg.tool_calls:
                    # Handle both dict and object formats
                    if isinstance(call, dict):
                        func_name = call.get('function', {}).get('name', '')
                        func_args = call.get('function', {}).get('arguments', '')
                    else:
                        func_name = call.function.name if hasattr(call.function, 'name') else ''
                        func_args = call.function.arguments if hasattr(call.function, 'arguments') else ''

                    if func_name:
                        tools_used.append(func_name)

                    # Extract SQL from run_sql_query tool call
                    if func_name == 'run_sql_query' and func_args:
                        try:
                            args = json.loads(func_args) if isinstance(func_args, str) else func_args
                            if 'sql' in args:
                                sql_query = args['sql']
                                print("[SQL Extracted from tool call]")
                            elif isinstance(args, dict):
                                print(f"[Debug] Tool args keys: {list(args.keys())}")
                        except Exception as e:
                            print(f"[Debug] Error parsing tool args: {str(e)}")

    # If SQL not found in tool calls, try to extract from response text
    if not sql_query:
        # Try to find SQL in code blocks first
        sql_blocks = re.findall(r'```sql\s+(.*?)```', content, re.DOTALL | re.IGNORECASE)
        if sql_blocks:
            sql_query = sql_blocks[-1].strip()  # Take last SQL block
        else:
            # Try to find SQL without code block markers
            sql_blocks = re.findall(r'```\s*(SELECT.*?)```', content, re.DOTALL | re.IGNORECASE)
            if sql_blocks:
                sql_query = sql_blocks[-1].strip()
            else:
                # Look for raw SELECT statement
                sql_match = re.search(r'(SELECT\s+.*?(?:;|\Z))', content, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    sql_query = sql_match.group(1).strip()

    if not sql_query:
        print(f"\n[ERROR] Could not extract SQL from response or tool calls")
        print(f"Tools used: {tools_used}")
        print(f"Response content:\n{content[:500]}\n")
        return {
            "error": "Could not extract SQL from agent response",
            "raw_response": content,
            "tools_used": list(set(tools_used))
        }

    result = {
        "sql": sql_query,
        "reasoning": content[:500] if len(content) > 500 else content,  # First 500 chars
        "tools_used": list(set(tools_used)),  # Unique tool names
        "tool_count": len(tools_used),
    }

    print(f"[SQL Generated]\n  Query: {sql_query[:100]}...")
    print(f"[Tools Used] {len(tools_used)} tool calls: {', '.join(set(tools_used))}")

    if execute:
        print(f"[2/3] Executing SQL...")
        sql_tool = SQLTools(db_url=db_url)

        try:
            result_json = sql_tool.run_sql_query(sql_query, limit=None)
            results = json.loads(result_json)
            result["results"] = results
            result["row_count"] = len(results)
            print(f"  Found {len(results)} records")
        except Exception as e:
            print(f"  Error: {e}")
            result["error"] = str(e)
            result["results"] = []
            return result

    if return_insights and execute and result.get("results"):
        print(f"[3/3] Generating insights...")
        insight_prompt = f"""Based on query results, provide brief summary:

Question: {question}
Results: {len(result['results'])} records

Sample: {json.dumps(result['results'][:3], indent=2, default=str)}

Provide: 1) One-sentence summary 2) 2-3 key findings. Be concise."""

        insight_response = dash.run(insight_prompt, stream=False)
        result["insights"] = insight_response.content if hasattr(insight_response, 'content') else str(insight_response)

    return result

if __name__ == "__main__":
    dash.print_response("List all tables in the database", stream=True)
