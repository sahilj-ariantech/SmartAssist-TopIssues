import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import psycopg2
from psycopg2 import sql
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

try:
    from .env_loader import load_local_env
    from .db_connection import get_db_config
except ImportError:
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from env_loader import load_local_env
    from db_connection import get_db_config

load_local_env(str(Path(__file__).resolve().parent / ".env"))

CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor"}
GENERIC_ISSUE_TERMS = {
    "issue",
    "issues",
    "problem",
    "problems",
    "system",
    "application",
    "app",
    "data",
    "customer",
    "users",
    "support",
    "team",
}
ISSUE_HINT_WORDS = {
    "not",
    "unable",
    "failed",
    "error",
    "errors",
    "blank",
    "pending",
    "missing",
    "delay",
    "showing",
    "reflecting",
    "updated",
    "received",
    "working",
    "completed",
}
ACRONYM_MAP = {"Otp": "OTP", "Cxp": "CXP", "Sa": "SA", "Dse": "DSE"}
ISSUE_PATTERNS = [
    "otp not received",
    "unable to export",
    "not matching",
    "mismatch",
    "inaccurate",
    "not updating",
    "not visible",
    "not getting completed",
    "not completed",
    "unable to",
    "not able to",
    "not working",
    "not showing",
    "not reflecting",
    "not reflected",
    "not updated",
    "missing from",
    "missing in",
    "partially visible",
    "partly visible",
    "deactivate account",
    "account deactivation",
    "not received",
    "error",
    "failed",
    "blank",
]
ISSUE_PATTERN_TITLE_MAP = {
    "otp not received": "OTP Not Received",
    "not getting completed": "Not Completed",
    "not completed": "Not Completed",
    "unable to": "Not Completed",
    "not able to": "Not Completed",
    "not working": "Not Working",
    "not showing": "Not Showing",
    "not reflecting": "Not Reflecting",
    "not reflected": "Not Reflecting",
    "not updating": "Data Not Updating",
    "not updated": "Not Updated",
    "not visible": "Records Not Visible",
    "unable to export": "Unable To Export",
    "not matching": "Data Mismatch",
    "mismatch": "Data Mismatch",
    "inaccurate": "Data Mismatch",
    "missing from": "Not Reflecting",
    "missing in": "Not Reflecting",
    "partially visible": "Partially Visible",
    "partly visible": "Partially Visible",
    "deactivate account": "Account Deactivation",
    "account deactivation": "Account Deactivation",
    "not received": "Not Received",
    "error": "Errors",
    "failed": "Failures",
    "blank": "Blank Data",
}
FILLER_TERMS = {
    "dear",
    "sir",
    "madam",
    "hello",
    "hi",
    "thanks",
    "thank",
    "regards",
    "kindly",
    "please",
    "request",
}
SUBJECT_PATTERNS = [
    ("smart assist", "Smart Assist"),
    ("qualified lead", "Qualified Lead"),
    ("lead transfer", "Lead Transfer"),
    ("transfer", "Lead Transfer"),
    ("follow up", "Follow Ups"),
    ("test drive", "Test Drive"),
    ("enquiry call", "Enquiry Calls"),
    ("enquiry", "Enquiry"),
    ("calls", "Calls"),
    ("call", "Calls"),
    ("otp", "OTP"),
    ("cxp", "CXP System"),
    ("vehicle", "Vehicle"),
    ("lead", "Lead"),
]
DOMAIN_SUBJECT_WORDS = {
    "smart",
    "assist",
    "lead",
    "transfer",
    "follow",
    "test",
    "drive",
    "enquiry",
    "call",
    "calls",
    "otp",
    "cxp",
    "vehicle",
    "booking",
    "ticket",
    "app",
    "application",
    "login",
    "password",
    "status",
    "qualified",
}
ISSUE_CATEGORY_LABELS = {
    "visibility": "Records Not Visible",
    "partial_visibility": "Partially Visible",
    "completion": "Unable To Complete",
    "not_received": "Not Received",
    "not_updating": "Data Not Updating",
    "data_mismatch": "Data Mismatch",
    "export": "Unable To Export",
    "not_working": "Not Working",
    "account_deactivation": "Account Deactivation",
    "errors": "System Errors",
}
NEGATIVE_TITLE_HINTS = {
    "not",
    "unable",
    "error",
    "errors",
    "fail",
    "failed",
    "missing",
    "blank",
}
DISALLOWED_GROUP_TITLES = {
    "general",
    "general issue",
    "general issues",
    "common",
    "common issue",
    "common issues",
    "other",
    "other issue",
    "other issues",
    "misc",
    "misc issue",
    "misc issues",
    "miscellaneous",
    "miscellaneous issue",
    "miscellaneous issues",
}
SUPPORTED_DATE_FILTERS = {
    "DAY",
    "YESTERDAY",
    "WEEK",
    "LAST_WEEK",
    "MTD",
    "LAST_MONTH",
    "QTD",
    "LAST_QUARTER",
    "SIX_MONTH",
    "YTD",
}
SUBJECT_SCORE_WEIGHTS = {
    "smart assist": 0.6,
    "cxp": 0.6,
}
ALLOWED_CLASS_FILTER_COLUMNS = {"category", "priority", "subcategory"}


def parse_args() -> argparse.Namespace:
    db_config = get_db_config()
    parser = argparse.ArgumentParser(
        description=(
            "Fetch issue descriptions from Postgres, group similar issues using ML, "
            "and print top issue groups."
        )
    )
    parser.add_argument("--db-name", default=db_config["DB_NAME"])
    parser.add_argument("--db-user", default=db_config["DB_USER"])
    parser.add_argument("--db-password", default=db_config["DB_PASSWORD"])
    parser.add_argument("--db-host", default=db_config["DB_HOST"])
    parser.add_argument("--db-port", type=int, default=int(db_config["DB_PORT"]))
    parser.add_argument("--table", default="Issues")
    parser.add_argument("--column", default="description")
    parser.add_argument("--subject-column", default=os.getenv("ISSUES_SUBJECT_COLUMN", "subject"))
    parser.add_argument("--dealer-column", default=os.getenv("ISSUES_DEALER_COLUMN", "dealer_name"))
    parser.add_argument("--ticket-id-column", default=os.getenv("ISSUES_TICKET_ID_COLUMN", "ticket_id"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-clusters", type=int, default=15)
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--date-column", default="date_reported")
    parser.add_argument("--date-filter", default=None)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--field-name", default=None)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--min-description-length",
        type=int,
        default=5,
        help="Drop descriptions shorter than this number of characters after normalization.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def is_generic_group_label(label: str) -> bool:
    return normalize_text(label).lower() in DISALLOWED_GROUP_TITLES


def infer_subject_from_text(text: str, max_words: int = 2) -> str:
    scrubbed = re.sub(r"\S+@\S+", " ", text.lower())
    scrubbed = re.sub(r"https?://\S+|www\.\S+", " ", scrubbed)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", scrubbed)

    selected: List[str] = []
    for token in tokens:
        if len(token) <= 2:
            continue
        if (
            token in CUSTOM_STOP_WORDS
            or token in FILLER_TERMS
            or token in ISSUE_HINT_WORDS
            or token in GENERIC_ISSUE_TERMS
        ):
            continue
        selected.append(token)
        if len(selected) >= max_words:
            break

    if not selected:
        return ""
    return apply_acronyms(" ".join(selected).title())


def canonical_title_tokens(value: str) -> set[str]:
    tokens: List[str] = []
    for raw_token in re.findall(r"[a-zA-Z]+", value.lower()):
        token = raw_token
        for suffix in ("ization", "isation", "ation", "tion", "ing", "ate", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[: -len(suffix)]
                break
        if len(token) <= 2 or token in CUSTOM_STOP_WORDS:
            continue
        tokens.append(token)
    return set(tokens)


def is_redundant_subject_issue(subject: str, issue_title: str) -> bool:
    subject_tokens = canonical_title_tokens(subject)
    issue_tokens = canonical_title_tokens(issue_title)
    if not subject_tokens or not issue_tokens:
        return False
    overlap = subject_tokens & issue_tokens
    if not overlap:
        return False
    minimum_len = min(len(subject_tokens), len(issue_tokens))
    return len(overlap) >= 2 or len(overlap) >= minimum_len


def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def compact_title(title: str, fallback_text: str = "") -> str:
    normalized = normalize_text(title)
    if not normalized:
        normalized = normalize_text(fallback_text)

    normalized = re.sub(r"\S+@\S+", "", normalized)
    normalized = normalize_text(normalized)
    if not normalized:
        return "Reported Concern"

    if " - " in normalized:
        left, right = split_title(normalized)
        left = limit_words(left, max_words=3)
        right = limit_words(right, max_words=4)
        normalized = f"{left} - {right}" if right else left
    else:
        normalized = limit_words(normalized, max_words=6)

    if len(normalized) > 56:
        normalized = normalized[:56].rstrip()
        if " " in normalized:
            normalized = normalized.rsplit(" ", 1)[0]
    return normalize_text(normalized)


def normalize_class_filter_value(raw_value: str) -> str:
    normalized_value = raw_value.strip()
    if (
        len(normalized_value) >= 2
        and normalized_value[0] == normalized_value[-1]
        and normalized_value[0] in {"'", '"'}
    ):
        normalized_value = normalized_value[1:-1].strip()
    if not normalized_value:
        raise ValueError("class_name filter value is required.")
    return normalized_value


def parse_class_name_filter(
    class_name: Optional[str],
    field_name: Optional[str],
) -> Optional[Tuple[str, str]]:
    raw_class_name = "" if class_name is None else class_name.strip()
    raw_field_name = "" if field_name is None else field_name.strip()

    if not raw_class_name and not raw_field_name:
        return None
    if raw_field_name and not raw_class_name:
        raise ValueError("class_name is required when field_name is provided.")

    if raw_field_name:
        normalized_field_name = raw_field_name.lower()
        if normalized_field_name not in ALLOWED_CLASS_FILTER_COLUMNS:
            allowed = ", ".join(sorted(ALLOWED_CLASS_FILTER_COLUMNS))
            raise ValueError(f"field_name must be one of: {allowed}.")
        return normalized_field_name, normalize_class_filter_value(raw_class_name)

    # Backward compatibility for older input style: class_name=\"column=value\".
    if "=" in raw_class_name:
        column, value = raw_class_name.split("=", 1)
        normalized_column = column.strip().lower()
        if normalized_column not in ALLOWED_CLASS_FILTER_COLUMNS:
            allowed = ", ".join(sorted(ALLOWED_CLASS_FILTER_COLUMNS))
            raise ValueError(f"field_name must be one of: {allowed}.")
        return normalized_column, normalize_class_filter_value(value)

    allowed = ", ".join(sorted(ALLOWED_CLASS_FILTER_COLUMNS))
    raise ValueError(
        f"field_name is required when class_name is provided. Use one of: {allowed}."
    )


def fetch_issue_records(
    conn: psycopg2.extensions.connection,
    table: str,
    column: str,
    subject_column: str,
    dealer_column: str,
    ticket_id_column: str,
    min_length: int,
    date_column: str,
    start_date: Optional[date],
    end_date: Optional[date],
    class_filter: Optional[Tuple[str, str]],
) -> List[Dict[str, str]]:
    conditions = [
        sql.SQL("{column} IS NOT NULL").format(column=sql.Identifier(column)),
        sql.SQL("BTRIM({column}) <> ''").format(column=sql.Identifier(column)),
    ]
    params: List[Any] = []
    if start_date and end_date:
        conditions.append(
            sql.SQL("{date_column}::date BETWEEN %s AND %s").format(
                date_column=sql.Identifier(date_column)
            )
        )
        params.extend([start_date, end_date])
    elif start_date:
        conditions.append(
            sql.SQL("{date_column}::date >= %s").format(date_column=sql.Identifier(date_column))
        )
        params.append(start_date)
    elif end_date:
        conditions.append(
            sql.SQL("{date_column}::date <= %s").format(date_column=sql.Identifier(date_column))
        )
        params.append(end_date)
    if class_filter:
        class_column, class_value = class_filter
        conditions.append(
            sql.SQL("{class_column} = %s").format(class_column=sql.Identifier(class_column))
        )
        params.append(class_value)

    query = sql.SQL(
        "SELECT {column}, {subject_column}, {dealer_column}, {ticket_id_column} "
        "FROM {table} WHERE {conditions}"
    ).format(
        column=sql.Identifier(column),
        subject_column=sql.Identifier(subject_column),
        dealer_column=sql.Identifier(dealer_column),
        ticket_id_column=sql.Identifier(ticket_id_column),
        table=sql.Identifier(table),
        conditions=sql.SQL(" AND ").join(conditions),
    )

    with conn.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()

    cleaned: List[Dict[str, str]] = []
    for row in rows:
        description = normalize_text(str(row[0]))
        subject = normalize_text(str(row[1])) if row[1] is not None else ""
        dealer_name = normalize_text(str(row[2])) if row[2] is not None else ""
        ticket_id = normalize_text(str(row[3])) if row[3] is not None else ""
        if len(description) >= min_length:
            model_text = f"{subject}. {description}" if subject else description
            cleaned.append(
                {
                    "description": description,
                    "subject": subject,
                    "dealer_name": dealer_name,
                    "ticket_id": ticket_id,
                    "model_text": model_text,
                }
            )
    return cleaned


def fetch_distinct_dealer_count(
    conn: psycopg2.extensions.connection,
    table: str,
    dealer_column: str,
    date_column: str,
    start_date: Optional[date],
    end_date: Optional[date],
    class_filter: Optional[Tuple[str, str]],
) -> int:
    conditions = [
        sql.SQL("{dealer_column} IS NOT NULL").format(dealer_column=sql.Identifier(dealer_column)),
        sql.SQL("BTRIM({dealer_column}) <> ''").format(dealer_column=sql.Identifier(dealer_column)),
    ]
    params: List[Any] = []
    if start_date and end_date:
        conditions.append(
            sql.SQL("{date_column}::date BETWEEN %s AND %s").format(
                date_column=sql.Identifier(date_column)
            )
        )
        params.extend([start_date, end_date])
    elif start_date:
        conditions.append(
            sql.SQL("{date_column}::date >= %s").format(date_column=sql.Identifier(date_column))
        )
        params.append(start_date)
    elif end_date:
        conditions.append(
            sql.SQL("{date_column}::date <= %s").format(date_column=sql.Identifier(date_column))
        )
        params.append(end_date)
    if class_filter:
        class_column, class_value = class_filter
        conditions.append(
            sql.SQL("{class_column} = %s").format(class_column=sql.Identifier(class_column))
        )
        params.append(class_value)

    query = sql.SQL(
        "SELECT COUNT(DISTINCT {dealer_column}) FROM {table} WHERE {conditions}"
    ).format(
        dealer_column=sql.Identifier(dealer_column),
        table=sql.Identifier(table),
        conditions=sql.SQL(" AND ").join(conditions),
    )

    with conn.cursor() as cursor:
        cursor.execute(query, params)
        row = cursor.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def choose_cluster_count(embeddings: np.ndarray, max_clusters: int) -> int:
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        return 1

    upper = min(max_clusters, n_samples - 1)
    if upper < 2:
        return 1

    eval_embeddings = embeddings
    if n_samples > 2000:
        rng = np.random.default_rng(42)
        sampled_idx = rng.choice(n_samples, size=2000, replace=False)
        eval_embeddings = embeddings[sampled_idx]

    best_k = 2
    best_score = -1.0

    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(eval_embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(eval_embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def shift_months(input_date: date, month_delta: int) -> date:
    month_index = (input_date.year * 12 + input_date.month - 1) + month_delta
    year = month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1)


def quarter_start(input_date: date) -> date:
    month = ((input_date.month - 1) // 3) * 3 + 1
    return date(input_date.year, month, 1)


def normalize_filter_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    if not normalized:
        return None

    aliases = {
        "TODAY": "DAY",
        "THIS_WEEK": "WEEK",
        "THIS_MONTH": "MTD",
        "THIS_QUARTER": "QTD",
        "LAST_6_MONTHS": "SIX_MONTH",
        "LAST_SIX_MONTHS": "SIX_MONTH",
        "THIS_YEAR": "YTD",
    }
    return aliases.get(normalized, normalized)


def resolve_date_range(
    date_filter: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
) -> Tuple[Optional[date], Optional[date], str]:
    normalized_filter = normalize_filter_name(date_filter)
    if not normalized_filter:
        raise ValueError("date_filter is required.")

    if start_date is not None or end_date is not None:
        raise ValueError("start_date and end_date are not supported. Send only date_filter.")

    if normalized_filter not in SUPPORTED_DATE_FILTERS:
        raise ValueError(
            "Unsupported date_filter. Use one of: "
            "DAY, YESTERDAY, WEEK, LAST_WEEK, MTD, LAST_MONTH, "
            "QTD, LAST_QUARTER, SIX_MONTH, YTD."
        )

    today = date.today()

    if normalized_filter == "DAY":
        return today, today, "Today"
    if normalized_filter == "YESTERDAY":
        day = today - timedelta(days=1)
        return day, day, "Yesterday"
    if normalized_filter == "WEEK":
        start = today - timedelta(days=today.weekday())
        return start, today, "This Week"
    if normalized_filter == "LAST_WEEK":
        this_week_start = today - timedelta(days=today.weekday())
        start = this_week_start - timedelta(days=7)
        end = this_week_start - timedelta(days=1)
        return start, end, "Last Week"
    if normalized_filter == "MTD":
        start = date(today.year, today.month, 1)
        return start, today, "This Month"
    if normalized_filter == "LAST_MONTH":
        this_month_start = date(today.year, today.month, 1)
        end = this_month_start - timedelta(days=1)
        start = date(end.year, end.month, 1)
        return start, end, "Last Month"
    if normalized_filter == "QTD":
        start = quarter_start(today)
        return start, today, "This Quarter"
    if normalized_filter == "LAST_QUARTER":
        this_q_start = quarter_start(today)
        end = this_q_start - timedelta(days=1)
        start = quarter_start(end)
        return start, end, "Last Quarter"
    if normalized_filter == "SIX_MONTH":
        month_start = date(today.year, today.month, 1)
        start = shift_months(month_start, -5)
        return start, today, "Last 6 Months"
    if normalized_filter == "YTD":
        # Rolling 365-day window including today.
        start = today - timedelta(days=364)
        return start, today, "Last 365 Days"

    raise ValueError("Invalid date_filter provided.")


def cluster_descriptions(
    descriptions: Sequence[str], encoder: SentenceTransformer, max_clusters: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = encoder.encode(
        list(descriptions),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings)

    k = choose_cluster_count(embeddings, max_clusters=max_clusters)
    if k == 1:
        labels = np.zeros(len(descriptions), dtype=int)
        centroids = np.mean(embeddings, axis=0, keepdims=True)
        return labels, embeddings, centroids

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, embeddings, kmeans.cluster_centers_


def extract_top_terms(
    cluster_texts: Sequence[str], ngram_range: Tuple[int, int], top_k: int
) -> List[str]:
    vectorizer = TfidfVectorizer(
        stop_words=list(CUSTOM_STOP_WORDS),
        ngram_range=ngram_range,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        matrix = vectorizer.fit_transform(cluster_texts)
    except ValueError:
        return []

    scores = np.asarray(matrix.sum(axis=0)).ravel()
    if scores.size == 0:
        return []

    terms = vectorizer.get_feature_names_out()
    top_indices = scores.argsort()[::-1]
    ranked_terms: List[str] = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        term = terms[idx].strip()
        if not term:
            continue
        ranked_terms.append(term)
        if len(ranked_terms) == top_k:
            break
    return ranked_terms


def apply_acronyms(title: str) -> str:
    updated = title
    for bad, good in ACRONYM_MAP.items():
        updated = updated.replace(bad, good)
    return updated


def detect_common_issue_phrase(cluster_texts: Sequence[str]) -> str:
    lowered = [text.lower() for text in cluster_texts]
    best_phrase = ""
    best_count = 0
    for phrase in ISSUE_PATTERNS:
        count = sum(phrase in text for text in lowered)
        if count > best_count:
            best_count = count
            best_phrase = phrase
    return best_phrase if best_count > 0 else ""


def detect_common_subject(cluster_texts: Sequence[str], min_count: int = 2) -> str:
    lowered = [text.lower() for text in cluster_texts]
    best_label = ""
    best_score = 0.0
    best_count = 0
    for pattern, label in SUBJECT_PATTERNS:
        count = sum(pattern in text for text in lowered)
        if count <= 0:
            continue
        weighted_score = count * SUBJECT_SCORE_WEIGHTS.get(pattern, 1.0)
        if weighted_score > best_score or (weighted_score == best_score and count > best_count):
            best_score = weighted_score
            best_count = count
            best_label = label
    return best_label if best_count >= min_count else ""


def build_group_title(
    cluster_texts: Sequence[str], cluster_subjects: Sequence[str], fallback_text: str
) -> str:
    source_texts = list(cluster_texts) if cluster_texts else [fallback_text]
    is_singleton = len(source_texts) == 1

    issue_phrase = detect_common_issue_phrase(source_texts)
    if not issue_phrase:
        phrase_candidates = extract_top_terms(source_texts, ngram_range=(2, 4), top_k=15)
        for phrase in phrase_candidates:
            words = set(phrase.split())
            if words & ISSUE_HINT_WORDS:
                issue_phrase = phrase
                break
    issue_title = ISSUE_PATTERN_TITLE_MAP.get(issue_phrase, issue_phrase)
    if not issue_title:
        issue_title = ISSUE_CATEGORY_LABELS.get(issue_category(" ".join(source_texts[:3])), "")

    normalized_subjects = [normalize_text(text) for text in cluster_subjects if normalize_text(text)]
    min_subject_hits = 1 if is_singleton else 2
    subject = detect_common_subject(normalized_subjects, min_count=min_subject_hits)
    if not subject and normalized_subjects:
        subject = detect_common_subject(source_texts, min_count=min_subject_hits)
    allow_subject_guess = (not is_singleton) or (not issue_title)
    if not subject and allow_subject_guess:
        subject_candidates: List[str] = []
        if normalized_subjects:
            subject_candidates.extend(
                extract_top_terms(normalized_subjects, ngram_range=(1, 2), top_k=12)
            )
        if not subject_candidates:
            subject_candidates = extract_top_terms(source_texts, ngram_range=(2, 2), top_k=12)
            subject_candidates.extend(extract_top_terms(source_texts, ngram_range=(1, 1), top_k=12))
        lowered_cluster_texts = [text.lower() for text in source_texts]
        for term in subject_candidates:
            words = term.split()
            if len(words) == 1 and (len(words[0]) <= 2 or words[0] in {"sa", "cxp"}):
                continue
            meaningful_words = [
                word
                for word in words
                if word not in FILLER_TERMS
                and word not in GENERIC_ISSUE_TERMS
                and word not in CUSTOM_STOP_WORDS
            ]
            if not meaningful_words:
                continue
            if any(word in FILLER_TERMS for word in words):
                continue
            if any(word in ISSUE_HINT_WORDS for word in meaningful_words):
                continue
            if all(word in GENERIC_ISSUE_TERMS for word in meaningful_words):
                continue
            term_count = sum(term.lower() in text for text in lowered_cluster_texts)
            if len(lowered_cluster_texts) > 1 and term_count < 2:
                continue
            subject = " ".join(meaningful_words[:3])
            break
    if not subject and not issue_title:
        subject = infer_subject_from_text(fallback_text)
    if subject and is_generic_group_label(subject):
        subject = ""

    if subject and issue_title:
        title = (
            issue_title
            if subject.lower() in issue_title.lower() or is_redundant_subject_issue(subject, issue_title)
            else f"{subject} - {issue_title}"
        )
    elif issue_title:
        title = issue_title
    elif subject:
        title = subject
    else:
        title = infer_subject_from_text(fallback_text) or "Reported Concern"

    return compact_title(apply_acronyms(title.title()), fallback_text=fallback_text)


def nearest_example_text(
    indices: Sequence[int],
    embeddings: np.ndarray,
    centroid: np.ndarray,
    descriptions: Sequence[str],
) -> str:
    cluster_vectors = embeddings[list(indices)]
    sims = cluster_vectors @ centroid
    best_local_idx = int(np.argmax(sims))
    best_global_idx = indices[best_local_idx]
    return descriptions[best_global_idx]


def split_title(title: str) -> Tuple[str, str]:
    if " - " not in title:
        return title.strip(), ""
    subject, issue = title.split(" - ", 1)
    return subject.strip(), issue.strip()


def normalize_subject_label(subject: str) -> str:
    lowered = normalize_text(subject).lower()
    if not lowered:
        return ""
    if "@" in lowered:
        return ""
    tokens = re.findall(r"[a-zA-Z]+", lowered)
    if len(tokens) > 4:
        return ""
    if tokens and any(token in ISSUE_HINT_WORDS for token in tokens):
        return ""
    if lowered in {"sa", "smart assist", "smartassist"}:
        return "Smart Assist"
    if lowered in {"cxp", "cxp system"}:
        return "CXP System"
    for pattern, label in SUBJECT_PATTERNS:
        if pattern in lowered:
            return label
    if is_generic_group_label(lowered):
        return ""
    return apply_acronyms(subject.title())


def issue_category(issue_label: str) -> str:
    lowered = issue_label.lower()
    if (
        "deactivate account" in lowered
        or ("deactivation" in lowered and "account" in lowered)
        or ("left" in lowered and "organization" in lowered)
    ):
        return "account_deactivation"
    if ("unable to export" in lowered) or ("export" in lowered and ("unable" in lowered or "not able" in lowered)):
        return "export"
    if (
        "not matching" in lowered
        or "mismatch" in lowered
        or "inaccurate" in lowered
        or ("shows 0" in lowered and "shows" in lowered)
    ):
        return "data_mismatch"
    if "not updating" in lowered:
        return "not_updating"
    if "not visible" in lowered:
        return "visibility"
    if (
        "not showing" in lowered
        or "not reflecting" in lowered
        or "not updated" in lowered
        or "blank" in lowered
    ):
        return "visibility"
    if "partially visible" in lowered or "partly visible" in lowered:
        return "partial_visibility"
    if "not completed" in lowered or "unable to" in lowered:
        return "completion"
    if "not working" in lowered:
        return "not_working"
    if "not received" in lowered:
        return "not_received"
    if "error" in lowered or "fail" in lowered:
        return "errors"
    return lowered or "other"


def finalize_group_title(title: str, fallback_text: str = "") -> str:
    normalized = normalize_text(title)
    if not normalized or is_generic_group_label(normalized):
        inferred = infer_subject_from_text(fallback_text)
        normalized = inferred if inferred else "Reported Ticket"

    lower_title = normalized.lower()
    if is_generic_group_label(lower_title):
        inferred = infer_subject_from_text(fallback_text)
        if inferred:
            normalized = inferred
            lower_title = normalized.lower()
        else:
            return "Reported Ticket Issues"

    if lower_title.endswith(" issue") or lower_title.endswith(" issues"):
        return compact_title(normalized, fallback_text=fallback_text)

    known_labels = {label.lower() for label in ISSUE_CATEGORY_LABELS.values()}
    known_labels.update(label.lower() for label in ISSUE_PATTERN_TITLE_MAP.values())
    if " - " in normalized or lower_title in known_labels:
        return compact_title(normalized, fallback_text=fallback_text)

    has_negative_signal = any(hint in lower_title for hint in NEGATIVE_TITLE_HINTS)
    word_count = len([token for token in normalized.replace("-", " ").split() if token])

    if not has_negative_signal or word_count <= 2:
        normalized = f"{normalized} Issues"
    return compact_title(normalized, fallback_text=fallback_text)


def merge_similar_groups(groups: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: Dict[Tuple[str, str], Dict[str, object]] = {}
    for item in groups:
        raw_title = str(item["title"])
        subject_raw, issue_raw = split_title(raw_title)
        subject = normalize_subject_label(subject_raw)
        source_for_category = issue_raw or subject_raw or str(item["example"])
        category = issue_category(source_for_category)
        if not subject and category == "other":
            subject = infer_subject_from_text(str(item["example"]))
        subject_key = subject.lower() if subject else normalize_text(issue_raw).lower()
        key = (subject_key, category)

        if key not in merged:
            merged[key] = {
                "subject": subject,
                "category": category,
                "count": 0,
                "example": item["example"],
                "best_count": int(item["count"]),
                "dealer_names": set(item["dealer_names"]),
                "ticket_ids": list(item["ticket_ids"]),
                "ticket_id_set": set(item["ticket_ids"]),
            }

        bucket = merged[key]
        item_count = int(item["count"])
        bucket["count"] = int(bucket["count"]) + item_count
        bucket["dealer_names"].update(item["dealer_names"])
        for ticket_id in item["ticket_ids"]:
            if ticket_id not in bucket["ticket_id_set"]:
                bucket["ticket_id_set"].add(ticket_id)
                bucket["ticket_ids"].append(ticket_id)
        if item_count >= int(bucket["best_count"]):
            bucket["best_count"] = item_count
            bucket["example"] = item["example"]

    output: List[Dict[str, object]] = []
    for bucket in merged.values():
        category = str(bucket["category"])
        issue_label = ISSUE_CATEGORY_LABELS.get(category, "")
        if not issue_label and category not in {"other", ""}:
            issue_label = apply_acronyms(category.replace("_", " ").title())
        if not issue_label:
            issue_phrase = detect_common_issue_phrase([str(bucket["example"])])
            issue_label = ISSUE_PATTERN_TITLE_MAP.get(issue_phrase, "")

        subject = normalize_text(str(bucket["subject"]))
        if subject and issue_label:
            title = (
                issue_label
                if subject.lower() in issue_label.lower()
                or is_redundant_subject_issue(subject, issue_label)
                else f"{subject} - {issue_label}"
            )
        elif issue_label:
            title = issue_label
        else:
            title = subject or infer_subject_from_text(str(bucket["example"])) or "Reported Concern"
        title = finalize_group_title(title, fallback_text=str(bucket["example"]))
        output.append(
            {
                "count": int(bucket["count"]),
                "dealer_count": len(bucket["dealer_names"]),
                "title": title,
                "ticket_ids": bucket["ticket_ids"],
                "example": bucket["example"],
            }
        )

    output.sort(key=lambda item: int(item["count"]), reverse=True)
    for idx, item in enumerate(output, start=1):
        item["rank"] = idx
    return output


def summarize_groups(
    descriptions: Sequence[str],
    subjects: Sequence[str],
    dealer_names: Sequence[str],
    ticket_ids: Sequence[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    centroids: np.ndarray,
    top_n: int,
) -> Tuple[List[Dict[str, object]], int]:
    members: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        members[int(label)].append(idx)

    ranked_clusters = sorted(members.items(), key=lambda item: len(item[1]), reverse=True)

    raw_summary: List[Dict[str, object]] = []
    for label, indices in ranked_clusters:
        texts = [descriptions[i] for i in indices]
        subject_texts = [subjects[i] for i in indices]
        cluster_dealer_names = {dealer_names[i] for i in indices if dealer_names[i]}
        cluster_ticket_ids = [ticket_ids[i] for i in indices if ticket_ids[i]]
        representative = nearest_example_text(
            indices=indices,
            embeddings=embeddings,
            centroid=centroids[label],
            descriptions=descriptions,
        )
        title = build_group_title(texts, subject_texts, representative)
        raw_summary.append(
            {
                "cluster_id": label,
                "count": len(indices),
                "title": title,
                "example": representative,
                "dealer_names": cluster_dealer_names,
                "ticket_ids": cluster_ticket_ids,
            }
        )

    merged = merge_similar_groups(raw_summary)
    return merged[:top_n], len(merged)


def print_results(results: Sequence[Dict[str, object]], total_rows: int) -> None:
    print(f"Processed descriptions: {total_rows}")
    print()
    print("Top issue groups:")
    for item in results:
        print(f"{item['rank']}. {item['title']}  (count: {item['count']})")
        print(f"   Example: {item['example']}")


def validate_db_fields(
    db_name: Optional[str],
    db_user: Optional[str],
    db_password: Optional[str],
    db_host: Optional[str],
    db_port: Optional[int],
) -> None:
    required = {
        "DB_NAME": db_name,
        "DB_USER": db_user,
        "DB_PASSWORD": db_password,
        "DB_HOST": db_host,
        "DB_PORT": db_port,
    }
    missing = [k for k, v in required.items() if v in (None, "")]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(
            f"Missing DB connection fields: {missing_csv}. "
            "Pass them as CLI args or environment variables."
        )


def validate_db_config(args: argparse.Namespace) -> None:
    validate_db_fields(
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        db_host=args.db_host,
        db_port=args.db_port,
    )


def analyze_top_issues(
    *,
    db_name: str,
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: int,
    table: str,
    column: str,
    subject_column: str,
    dealer_column: str,
    ticket_id_column: str,
    top_n: int,
    max_clusters: int,
    min_description_length: int,
    date_column: str,
    date_filter: Optional[str],
    class_name: Optional[str],
    field_name: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
    encoder: SentenceTransformer,
) -> Dict[str, object]:
    validate_db_fields(
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        db_host=db_host,
        db_port=db_port,
    )
    if top_n < 1:
        raise ValueError("top_n must be greater than or equal to 1.")
    if max_clusters < 1:
        raise ValueError("max_clusters must be greater than or equal to 1.")
    if min_description_length < 1:
        raise ValueError("min_description_length must be greater than or equal to 1.")
    if not date_column or not date_column.strip():
        raise ValueError("date_column is required.")
    if not subject_column or not subject_column.strip():
        raise ValueError("subject_column is required.")
    if not dealer_column or not dealer_column.strip():
        raise ValueError("dealer_column is required.")
    if not ticket_id_column or not ticket_id_column.strip():
        raise ValueError("ticket_id_column is required.")

    resolved_start, resolved_end, resolved_filter = resolve_date_range(
        date_filter=date_filter,
        start_date=start_date,
        end_date=end_date,
    )
    class_filter = parse_class_name_filter(class_name=class_name, field_name=field_name)

    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
    )
    try:
        total_dealer_count = fetch_distinct_dealer_count(
            conn=conn,
            table=table,
            dealer_column=dealer_column,
            date_column=date_column,
            start_date=resolved_start,
            end_date=resolved_end,
            class_filter=class_filter,
        )
        records = fetch_issue_records(
            conn=conn,
            table=table,
            column=column,
            subject_column=subject_column,
            dealer_column=dealer_column,
            ticket_id_column=ticket_id_column,
            min_length=min_description_length,
            date_column=date_column,
            start_date=resolved_start,
            end_date=resolved_end,
            class_filter=class_filter,
        )
    finally:
        conn.close()

    if not records:
        return {
            "processed_descriptions": 0,
            "total_dealer_count": total_dealer_count,
            "total_issue_groups": 0,
            "applied_date_filter": resolved_filter,
            "start_date": resolved_start.isoformat() if resolved_start else None,
            "end_date": resolved_end.isoformat() if resolved_end else None,
            "top_issues": [],
        }

    descriptions = [row["description"] for row in records]
    subjects = [row["subject"] for row in records]
    dealer_names = [row["dealer_name"] for row in records]
    ticket_ids = [row["ticket_id"] for row in records]
    model_texts = [row["model_text"] for row in records]

    labels, embeddings, centroids = cluster_descriptions(
        descriptions=model_texts,
        encoder=encoder,
        max_clusters=max_clusters,
    )
    top_issues, total_issue_groups = summarize_groups(
        descriptions=descriptions,
        subjects=subjects,
        dealer_names=dealer_names,
        ticket_ids=ticket_ids,
        labels=labels,
        embeddings=embeddings,
        centroids=centroids,
        top_n=top_n,
    )
    return {
        "processed_descriptions": len(descriptions),
        "total_dealer_count": total_dealer_count,
        "total_issue_groups": total_issue_groups,
        "applied_date_filter": resolved_filter,
        "start_date": resolved_start.isoformat() if resolved_start else None,
        "end_date": resolved_end.isoformat() if resolved_end else None,
        "top_issues": top_issues,
    }


def main() -> None:
    args = parse_args()
    validate_db_config(args)
    encoder = SentenceTransformer(args.model_name)
    result = analyze_top_issues(
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        db_host=args.db_host,
        db_port=args.db_port,
        table=args.table,
        column=args.column,
        subject_column=args.subject_column,
        dealer_column=args.dealer_column,
        ticket_id_column=args.ticket_id_column,
        top_n=args.top_n,
        max_clusters=args.max_clusters,
        min_description_length=args.min_description_length,
        date_column=args.date_column,
        date_filter=args.date_filter,
        class_name=args.class_name,
        field_name=args.field_name,
        start_date=args.start_date,
        end_date=args.end_date,
        encoder=encoder,
    )
    print(f"Total issue groups found: {result['total_issue_groups']}")
    print_results(result["top_issues"], total_rows=result["processed_descriptions"])


if __name__ == "__main__":
    main()
