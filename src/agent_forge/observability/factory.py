from agent_forge.config import settings
from agent_forge.observability.sqlite_tracer import SQLiteTracer


def build_tracer(db_url: str | None = None) -> SQLiteTracer:
    return SQLiteTracer(db_url or settings.db_url)
