"""DriveMate's offline VOICE-T04 canned responses."""


async def generate_alert(ctx: dict, timeout_s: float = 2.0) -> str | None:
    return "Ești bine?" if ctx.get("lang") == "ro" else "Are you okay?"


async def generate_engagement(ctx: dict, timeout_s: float = 2.0) -> dict | None:
    if ctx.get("lang") == "ro":
        return {
            "kind": "trivia",
            "text": "Care este capitala Franței?",
            "answer": "Paris",
        }
    return {
        "kind": "trivia",
        "text": "What is the capital of France?",
        "answer": "Paris",
    }


async def grade_answer(
    question: str, expected: str, reply: str, timeout_s: float = 1.5
) -> dict | None:
    return {"correct": False, "feedback": "Thanks for answering."}


async def classify_reply(
    transcript: str, expecting: str, timeout_s: float = 1.5
) -> str | None:
    return "UNCLEAR"


async def summarize_trip(log: list[dict], timeout_s: float = 10.0) -> str | None:
    return "Thank you for driving with DriveMate."
