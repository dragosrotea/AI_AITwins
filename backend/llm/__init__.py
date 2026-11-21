"""DriveMate's language functions and checkpoint stubs."""

from .replies import classify_reply, generate_alert

__all__ = [
    "classify_reply",
    "generate_alert",
    "generate_engagement",
    "grade_answer",
    "summarize_trip",
]


async def generate_engagement(ctx: dict, timeout_s: float = 2.0) -> dict | None:
    return {
        "kind": "trivia",
        "text": "What is the capital of France?",
        "answer": "Paris",
    }


async def grade_answer(
    question: str, expected: str, reply: str, timeout_s: float = 1.5
) -> dict | None:
    return {"correct": False, "feedback": "Thanks for answering."}


async def summarize_trip(log: list[dict], timeout_s: float = 10.0) -> str | None:
    return "Thank you for driving with DriveMate."
