"""
Database URLs
=============

Build database connection URLs from environment variables.

Two separate connections:
  - target_db_url: SQL Server (the database the agent queries)
  - internal storage: SQLite + LanceDB (file-based, no URL needed)
"""

from os import getenv
from urllib.parse import quote_plus


def build_target_db_url() -> str:
    """Build SQLAlchemy URL for the target SQL Server database."""
    server = getenv("SQL_SERVER", ".")
    database = getenv("SQL_DATABASE", "ERPNextDB")
    driver = getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")

    # Build pyodbc connection string for Windows Authentication
    odbc_conn = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection=yes;"
    )
    return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_conn)}"


# Target DB: SQL Server (where the agent runs queries)
target_db_url = build_target_db_url()

# Keep backward-compatible name for imports that use `db_url`
db_url = target_db_url
