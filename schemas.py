"""
Dash Response Schemas for Non-Interactive Mode
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class DashSqlResponse(BaseModel):
    """Structured SQL response for non-interactive mode"""

    sql_query: str = Field(
        description="Complete T-SQL query to execute (REQUIRED, valid SQL Server syntax)"
    )

    tables_used: List[str] = Field(
        description="List of table names referenced in the query"
    )

    reasoning: str = Field(
        description="Brief explanation why this query answers the question (1-2 sentences)"
    )

    joins_explanation: Optional[str] = Field(
        default=None,
        description="If query uses JOINs, explain how the tables relate"
    )

    expected_result_type: str = Field(
        description="What type of data this returns (e.g. 'person records', 'sales totals')"
    )
