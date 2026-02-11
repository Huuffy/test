"""
Dash API
========

FastAPI server for the Dash text-to-SQL agent.

Run:
    uvicorn app.main:app --reload --port 8000
"""

from os import getenv

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from dash.agents import dash, reasoning_dash, dash_knowledge
from db import target_db_url

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DatabaseError, OperationalError

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Dash — Text-to-SQL Agent",
    description="A self-learning data agent for SQL Server",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQL Server engine for direct queries
engine = create_engine(target_db_url)


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    question: str
    reasoning: bool = False


class QueryResponse(BaseModel):
    question: str
    response: str


class ExecuteResponse(BaseModel):
    question: str
    response: str
    sql: str | None = None
    columns: list[str] = []
    results: list[dict] = []


class TableSchemaResponse(BaseModel):
    table_name: str
    columns: list[dict]


# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
def health_check():
    """Check system health: SQL Server connection + Ollama availability."""
    status = {"status": "ok", "sql_server": False, "ollama": False}

    # Test SQL Server
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["sql_server"] = True
    except Exception as e:
        status["sql_server_error"] = str(e)

    # Test Ollama
    try:
        import httpx
        ollama_url = getenv("OLLAMA_HOST", "http://localhost:11434")
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        status["ollama"] = resp.status_code == 200
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            status["ollama_models"] = models
    except Exception:
        status["ollama"] = False

    if not status["sql_server"] or not status["ollama"]:
        status["status"] = "degraded"

    return status


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Send a natural language question → get the agent's response (with SQL)."""
    agent = reasoning_dash if request.reasoning else dash
    try:
        run_response = agent.run(request.question)
        return QueryResponse(
            question=request.question,
            response=run_response.content,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute", response_model=ExecuteResponse)
def execute(request: QueryRequest):
    """Send a natural language question → get SQL + executed results."""
    agent = reasoning_dash if request.reasoning else dash
    try:
        run_response = agent.run(request.question)
        response_text = run_response.content

        # Try to extract SQL from the response (between ```sql ... ``` blocks)
        sql = None
        columns = []
        results = []

        if "```sql" in response_text:
            parts = response_text.split("```sql")
            if len(parts) > 1:
                sql_block = parts[1].split("```")[0].strip()
                sql = sql_block

                # Execute the extracted SQL
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(sql))
                        columns = list(result.keys())
                        rows = result.fetchall()
                        results = [dict(zip(columns, row)) for row in rows]
                except Exception:
                    pass  # SQL extraction/execution failed, return response without results

        return ExecuteResponse(
            question=request.question,
            response=response_text,
            sql=sql,
            columns=columns,
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables")
def list_tables():
    """List all tables in the SQL Server database."""
    try:
        insp = inspect(engine)
        tables = sorted(insp.get_table_names())
        return {"tables": tables, "count": len(tables)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema/{table_name}", response_model=TableSchemaResponse)
def get_schema(table_name: str):
    """Get schema for a specific table."""
    try:
        insp = inspect(engine)
        tables = insp.get_table_names()
        if table_name not in tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

        cols = insp.get_columns(table_name)
        columns = [
            {
                "name": c["name"],
                "type": str(c["type"]),
                "nullable": c.get("nullable", True),
            }
            for c in cols
        ]
        return TableSchemaResponse(table_name=table_name, columns=columns)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=getenv("API_HOST", "0.0.0.0"),
        port=int(getenv("API_PORT", "8000")),
        reload=getenv("RUNTIME_ENV", "dev") == "dev",
    )
