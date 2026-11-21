"""DriveMate's language functions and checkpoint stubs."""

from .engagement import generate_engagement, grade_answer
from .replies import classify_reply, generate_alert

__all__ = [
    "classify_reply",
    "generate_alert",
    "generate_engagement",
    "grade_answer",
    "summarize_trip",
]


async def summarize_trip(log: list[dict], timeout_s: float = 10.0) -> str | None:
    return "Thank you for driving with DriveMate."
