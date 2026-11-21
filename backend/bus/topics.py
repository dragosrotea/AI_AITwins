"""Topic pattern matching shared by the broker and the clients."""


def topic_matches(pattern: str, topic: str) -> bool:
    """Exact match, or prefix match when the pattern ends in '*' ('*' alone matches all)."""
    if pattern.endswith("*"):
        return topic.startswith(pattern[:-1])
    return pattern == topic
