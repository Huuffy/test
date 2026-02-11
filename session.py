"""
Database Session
================

SQLite database for agent state storage (replaces PostgresDb).
LanceDB is used for vector storage (configured in agents.py).
"""

from os import getenv
from pathlib import Path

from agno.db.sqlite import SqliteDb

# Store agent data in a local SQLite file
DATA_DIR = Path(getenv("DATA_DIR", Path(__file__).parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_DB_FILE = str(DATA_DIR / "dash_agent.db")
DB_ID = "dash-db"


def get_agent_db() -> SqliteDb:
    """Create a SqliteDb instance for agent state storage."""
    return SqliteDb(id=DB_ID, db_file=SQLITE_DB_FILE)
