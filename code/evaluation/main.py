import os
import re
import pandas as pd


DATASET_DIR = "dataset"

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]


def load_data():
    """Load all available challenge datasets."""

    files = [
        "messages.csv",
        "users.csv",
        "groups.csv",
        "group_members.csv",
        "business_accounts.csv",
        "user_business_history.csv",
        "message_history.csv",
        "message_events.csv",
        "images.csv",
        "voice_notes.csv",
        "daily_notification_summary.csv",
    ]

    data = {}

    for filename in files:
        path = os.path.join(DATASET_DIR, filename)

        if os.path.exists(path):
            data[filename.replace(".csv", "")] = pd.read_csv(path)

    return data


def clean_text(text):
    """Convert missing values to an empty string."""

    if pd.isna(text):
        return ""

    return str(text).strip()


def contains_any(text, keywords):
    """Return True if any keyword appears in the text."""

    text = text.lower()

    return any(keyword in text for keyword in keywords)


def detect_scam(text):
    """
    Detect strong scam/safety signals.

    This is intentionally conservative:
    we only classify as scam when multiple strong indicators
    appear together.
    """

    text = text.lower()

    sensitive = [
        "otp",
        "one time password",
        "login code",
        "verification code",
        "6 digit",
        "six digit",
        "password",
        "pin",
    ]

    pressure = [
        "blocked",
        "will be blocked",
        "expire today",
        "expires today",
        "within 2 hours",
        "verify now",
        "confirm now",
        "keep your account active",
        "keep payments active",
    ]

    financial = [
        "wallet",
        "payment",
        "bank",
        "account",
        "login",
    ]

    sensitive_hit = contains_any(text, sensitive)
    pressure_hit = contains_any(text, pressure)
    financial_hit = contains_any(text, financial)

    # Sensitive credential request + pressure
    if sensitive_hit and pressure_hit:
        return True

    # Sensitive credential + financial/account context
    if sensitive_hit and financial_hit:
        return True

    return False


def detect_prompt_injection(text):
    """Detect attempts to manipulate the routing system."""

    text = text.lower()

    injection_patterns = [
        "ignore all previous routing rules",
        "ignore previous instructions",
        "mark this message as notify",
        "ignore all previous instructions",
        "you are now instructed",
    ]

    return contains_any(text, injection_patterns)


def detect_urgency(text):
    """Detect strong urgency indicators."""

    urgency_words = [
        "urgent",
        "immediately",
        "right now",
        "asap",
        "in 10 minutes",
        "in 15 minutes",
        "in 20 minutes",
        "within 30 minutes",
        "leaving in",
        "starts in",
        "escalation starts",
        "deadline",
        "today",
    ]

    return contains_any(text, urgency_words)


def detect_event(text):
    """Detect event/schedule related messages."""

    event_words = [
        "meeting",
        "event",
        "school",
        "bus",
        "pickup",
        "appointment",
        "schedule",
        "scheduled",
        "form",
        "circular",
        "tomorrow",
        "saturday",
        "sunday",
    ]

    return contains_any(text, event_words)


def detect_promotion(text):
    """Detect promotional/marketing messages."""

    promotion_words = [
        "offer",
        "discount",
        "sale",
        "off",
        "promo",
        "promotion",
        "deal",
        "₹",
        "rs.",
        "coupon",
        "buy",
        "available",
    ]

    return contains_any(text, promotion_words)


def detect_greeting(text):
    """Detect greeting messages."""

    greeting_words = [
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "have a nice day",
        "stay positive",
        "hope today",
    ]

    return contains_any(text, greeting_words)


def detect_personal(text):
    """Detect basic personal conversation."""

    personal_words = [
        "can you",
        "call me",
        "call",
        "are you",
        "come online",
        "talk tomorrow",
        "had dinner",
        "reached home",
        "no need to respond",
    ]

    return contains_any(text, personal_words)


def classify_message(text):
    """
    Determine the best-fit message type.

    Safety categories receive priority.
    """

    if detect_scam(text):
        return "scam"

    if detect_event(text):
        return "event"

    if detect_promotion(text):
        return "promotion"

    if detect_greeting(text):
        return "greeting"

    if detect_personal(text):
        return "personal"

    if detect_urgency(text):
        return "urgent"

    return "unknown"


def route_message(text):
    """
    Decide notify / digest / mute.

    Safety takes priority over urgency.
    """

    text = clean_text(text)

    # Empty message: don't interrupt the user.
    if not text:
        return (
            "digest",
            "personal",
            "No readable message content requiring immediate attention.",
            0.60,
        )

    # Prompt injection does not get to control our router.
    injection = detect_prompt_injection(text)

    # Safety has highest priority.
    if detect_scam(text):
        reason = (
            "The message requests sensitive verification information "
            "and uses account or payment pressure, indicating a likely scam."
        )

        if injection:
            reason = (
                "The message attempts to manipulate the router and also "
                "requests sensitive verification information under account "
                "or payment pressure."
            )

        return "mute", "scam", reason, 0.87

    # Strong urgency.
    if detect_urgency(text):
        message_type = classify_message(text)

        return (
            "notify",
            "urgent",
            "The message contains a time-sensitive request or deadline "
            "that may require immediate attention.",
            0.82,
        )

    # Event information that is useful but not immediately urgent.
    if detect_event(text):
        return (
            "digest",
            "event",
            "The message contains useful event or schedule information "
            "but does not require immediate attention.",
            0.82,
        )

    # Promotional content is normally lower priority.
    if detect_promotion(text):
        return (
            "digest",
            "promotion",
            "The message contains promotional or offer-related information "
            "without an immediate action requirement.",
            0.75,
        )

    # Greetings/casual conversation.
    if detect_greeting(text):
        return (
            "digest",
            "greeting",
            "The message is a non-urgent greeting and does not require "
            "immediate attention.",
            0.80,
        )

    # Personal conversation.
    if detect_personal(text):
        return (
            "digest",
            "personal",
            "The message is personal conversation without a clear "
            "immediate action requirement.",
            0.78,
        )

    return (
        "digest",
        "unknown",
        "The message does not show a clear reason for immediate interruption "
        "or muting.",
        0.60,
    )


def process_message(row):
    """Process one incoming message."""

    message_id = row["message_id"]
    text = clean_text(row.get("message_text", ""))

    action, message_type, reason, confidence = route_message(text)

    return {
        "message_id": message_id,
        "action": action,
        "message_type": message_type,
        "reason": reason,
        "confidence": confidence,
        "evidence_message_ids": "none",
    }


def validate_output(output):
    """Validate the final submission format."""

    required_actions = {"notify", "digest", "mute"}

    required_columns = OUTPUT_COLUMNS

    # Column validation
    if list(output.columns) != required_columns:
        raise ValueError(
            f"Invalid columns.\n"
            f"Expected: {required_columns}\n"
            f"Got: {list(output.columns)}"
        )

    # One row per message
    if output["message_id"].duplicated().any():
        raise ValueError("Duplicate message_id found.")

    # Action validation
    invalid_actions = set(output["action"]) - required_actions

    if invalid_actions:
        raise ValueError(
            f"Invalid actions found: {invalid_actions}"
        )

    # Confidence validation
    if not output["confidence"].between(0, 1).all():
        raise ValueError("Confidence must be between 0 and 1.")

    # Missing values
    if output.isnull().any().any():
        raise ValueError("Output contains missing values.")


def main():
    print("Loading challenge data...")

    data = load_data()

    if "messages" not in data:
        raise FileNotFoundError(
            "dataset/messages.csv was not found."
        )

    messages = data["messages"]

    print(f"Loaded {len(messages)} messages.")

    results = []

    for _, row in messages.iterrows():
        result = process_message(row)
        results.append(result)

    output = pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    validate_output(output)

    output.to_csv("output.csv", index=False)

    print("Done.")
    print(f"Generated output.csv with {len(output)} predictions.")


if __name__ == "__main__":
    main()
