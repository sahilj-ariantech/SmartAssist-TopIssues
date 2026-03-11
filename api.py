import os
import sys
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

try:
    from .env_loader import load_local_env
    from .top_issues import analyze_top_issues
    from .db_connection import get_db_config
except ImportError:
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from env_loader import load_local_env
    from top_issues import analyze_top_issues
    from db_connection import get_db_config

load_local_env(str(Path(__file__).resolve().parent / ".env"))

APP_NAME = "Top Issues API"
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_TABLE = os.getenv("ISSUES_TABLE", "Issues")
DEFAULT_COLUMN = os.getenv("ISSUES_COLUMN", "description")
DEFAULT_SUBJECT_COLUMN = os.getenv("ISSUES_SUBJECT_COLUMN", "subject")
DEFAULT_DEALER_COLUMN = os.getenv("ISSUES_DEALER_COLUMN", "dealer_name")
DEFAULT_TICKET_ID_COLUMN = os.getenv("ISSUES_TICKET_ID_COLUMN", "ticket_id")
DEFAULT_DATE_COLUMN = os.getenv("ISSUES_DATE_COLUMN", "date_reported")
DEFAULT_TOP_N = int(os.getenv("TOP_N", "5"))
DEFAULT_MAX_CLUSTERS = int(os.getenv("MAX_CLUSTERS", "15"))
DEFAULT_MIN_DESCRIPTION_LENGTH = int(os.getenv("MIN_DESCRIPTION_LENGTH", "5"))

app = FastAPI(title=APP_NAME, version="1.0.0")
encoder = SentenceTransformer(MODEL_NAME)


def _db_config() -> Dict[str, Any]:
    config = get_db_config()
    return {
        "db_name": config["DB_NAME"],
        "db_user": config["DB_USER"],
        "db_password": config["DB_PASSWORD"],
        "db_host": config["DB_HOST"],
        "db_port": int(config["DB_PORT"]),
    }


def _serialize_issues(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for item in items:
        output.append(
            {
                "rank": item["rank"],
                "title": item["title"],
                "count": item["count"],
                "dealer_count": item["dealer_count"],
                "ticket_ids": item["ticket_ids"],
            }
        )
    return output


class TopIssuesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    date_filter: str
    class_name: Optional[str] = None
    field_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("Field_name", "field_name"),
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/top-issues")
def get_top_issues(payload: TopIssuesRequest) -> Dict[str, object]:
    try:
        result = analyze_top_issues(
            **_db_config(),
            table=DEFAULT_TABLE,
            column=DEFAULT_COLUMN,
            subject_column=DEFAULT_SUBJECT_COLUMN,
            dealer_column=DEFAULT_DEALER_COLUMN,
            ticket_id_column=DEFAULT_TICKET_ID_COLUMN,
            date_column=DEFAULT_DATE_COLUMN,
            date_filter=payload.date_filter,
            class_name=payload.class_name,
            field_name=payload.field_name,
            start_date=None,
            end_date=None,
            top_n=DEFAULT_TOP_N,
            max_clusters=DEFAULT_MAX_CLUSTERS,
            min_description_length=DEFAULT_MIN_DESCRIPTION_LENGTH,
            encoder=encoder,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "applied_date_filter": result["applied_date_filter"],
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "processed_descriptions": result["processed_descriptions"],
        "total_dealer_count": result["total_dealer_count"],
        "total_issue_groups": result["total_issue_groups"],
        "top_issues": _serialize_issues(result["top_issues"]),
    }
