"""Serve the same-origin explainability dashboard."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

DASHBOARD_ROOT = Path(__file__).parents[2] / "dashboard"
router = APIRouter(include_in_schema=False)


@router.get("/", response_class=FileResponse)
def dashboard() -> FileResponse:
    """Return the dashboard shell; data is loaded from the public API."""
    return FileResponse(DASHBOARD_ROOT / "index.html")
