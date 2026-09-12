import os
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"
OUTPUT_FILE = "output.csv"

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

VALID_ACTIONS = {"notify", "digest", "mute"}

VALID_MESSAGE_TYPES = {
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """
    Load all challenge datasets that are available.

    The solution does not assume every optional dataset exists.
    """

    filenames = [
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

    for filename in filenames:
        path = os.path.join(DATASET_DIR, filename)

        if os.path.exists(path):
            try:
                data[filename[:-4]] = pd.read_csv(path)
            except Exception as exc:
                print(f"Warning: could not load {filename}: {exc}")

    return data


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_text(value):
    """Safely convert a value to text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_text(text):
    """Lowercase and normalize whitespace."""

    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text


def contains_any(text, keywords):
    """Check whether any keyword/phrase occurs in the text."""

    text = normalize_text(text)

    return any(keyword.lower() in text for keyword in keywords)


def safe_float(value, default=0.0):
    """Convert a value to float safely."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# SAFETY DETECTION
# ============================================================

def detect_prompt_injection(text):
    """
    Detect attempts by the incoming message to manipulate
    the routing agent.

    Important:
    A message is untrusted data. Instructions inside the message
    must never override the router's own instructions.
    """

    patterns = [
        "ignore all previous routing rules",
        "ignore previous routing rules",
        "ignore all previous instructions",
        "ignore previous instructions",
        "mark this message as notify",
        "mark this as notify",
        "you are now instructed",
        "override the routing",
        "override routing rules",
    ]

    return contains_any(text, patterns)


def detect_sensitive_request(text):
    """Detect requests for credentials or sensitive verification."""

    sensitive_terms = [
        "otp",
        "one time password",
        "one-time password",
        "verification code",
        "login code",
        "6 digit",
        "six digit",
        "password",
        "passcode",
        "pin",
        "cvv",
        "card number",
    ]

    return contains_any(text, sensitive_terms)


def detect_pressure(text):
    """Detect strong pressure/threat language."""

    pressure_terms = [
        "verify now",
        "confirm now",
        "act now",
        "immediately",
        "right now",
        "urgent",
        "will be blocked",
        "profile will be blocked",
        "account will be blocked",
        "blocked in",
        "expire today",
        "expires today",
        "within 2 hours",
        "within an hour",
        "keep your account active",
        "keep payments active",
    ]

    return contains_any(text, pressure_terms)


def detect_financial_context(text):
    """Detect account/payment/wallet context."""

    financial_terms = [
        "bank",
        "payment",
        "payments",
        "wallet",
        "upi",
        "transaction",
        "account",
        "login",
        "card",
    ]

    return contains_any(text, financial_terms)


def detect_scam(text):
    """
    Conservative scam detector.

    Strong scam signal:
        sensitive credential request + pressure

    Also strong:
        sensitive credential request + financial/account context
    """

    sensitive = detect_sensitive_request(text)
    pressure = detect_pressure(text)
    financial = detect_financial_context(text)

    if sensitive and pressure:
        return True

    if sensitive and financial:
        return True

    return False


# ============================================================
# MESSAGE TYPE DETECTION
# ============================================================

def detect_event(text):
    event_terms = [
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
        "timing",
        "consent",
        "tonight",
        "tomorrow",
        "saturday",
        "sunday",
    ]

    return contains_any(text, event_terms)


def detect_promotion(text):
    promotion_terms = [
        "offer",
        "discount",
        "sale",
        "promo",
        "promotion",
        "deal",
        "coupon",
        "cashback",
        "off",
        "₹",
        "rs.",
        "buy",
        "shop",
        "limited time",
    ]

    return contains_any(text, promotion_terms)


def detect_greeting(text):
    greeting_terms = [
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "have a nice day",
        "stay positive",
        "hope today",
        "hello everyone",
        "hi everyone",
    ]

    return contains_any(text, greeting_terms)


def detect_forward(text):
    forward_terms = [
        "forward this",
        "please forward",
        "forward to family",
        "forward to friends",
        "forwarded",
    ]

    return contains_any(text, forward_terms)


def detect_payment(text):
    payment_terms = [
        "payment",
        "paid",
        "payment received",
        "invoice",
        "bill",
        "transaction",
        "refund",
        "receipt",
    ]

    return contains_any(text, payment_terms)


def detect_urgent(text):
    urgency_terms = [
        "urgent",
        "immediately",
        "right now",
        "asap",
        "act now",
        "in 10 minutes",
        "in 15 minutes",
        "in 20 minutes",
        "within 20 minutes",
        "within 30 minutes",
        "leaving in",
        "starts in",
        "escalation starts",
        "deadline",
        "before eod",
        "eod",
        "today",
        "now",
    ]

    return contains_any(text, urgency_terms)


def detect_personal(text):
    personal_terms = [
        "can you",
        "could you",
        "call me",
        "call",
        "are you free",
        "are you",
        "come online",
        "talk tomorrow",
        "reached home",
        "had dinner",
        "nothing urgent",
        "no need to respond",
    ]

    return contains_any(text, personal_terms)


def classify_message_type(text):
    """
    Determine message type.

    Safety categories get priority over ordinary categories.
    """

    if detect_scam(text):
        return "scam"

    if detect_payment(text):
        return "payment"

    if detect_urgent(text):
        return "urgent"

    if detect_event(text):
        return "event"

    if detect_promotion(text):
        return "promotion"

    if detect_forward(text):
        return "forward"

    if detect_greeting(text):
        return "greeting"

    if detect_personal(text):
        return "personal"

    return "unknown"


# ============================================================
# USER / HISTORY RETRIEVAL
# ============================================================

def get_user_history(data, user_id):
    """
    Retrieve only the historical records relevant to this user.
    """

    history = {}

    tables = [
        "message_history",
        "message_events",
        "user_business_history",
    ]

    for table_name in tables:
        table = data.get(table_name)

        if table is None or table.empty:
            history[table_name] = pd.DataFrame()
            continue

        if "user_id" in table.columns:
            history[table_name] = table[
                table["user_id"].astype(str) == str(user_id)
            ].copy()
        else:
            history[table_name] = pd.DataFrame()

    return history


def find_history_evidence(history):
    """
    Extract message IDs that may support the routing decision.

    We keep this deliberately conservative.
    """

    evidence = []

    for table_name in ["message_history", "message_events"]:
        table = history.get(table_name)

        if table is None or table.empty:
            continue

        possible_columns = [
            "message_id",
            "related_message_id",
            "source_message_id",
            "evidence_message_id",
        ]

        for column in possible_columns:
            if column in table.columns:
                for value in table[column].dropna().astype(str):
                    if value and value not in evidence:
                        evidence.append(value)

    return evidence


def history_contains_unwanted_behavior(history):
    """
    Look for previous user behavior indicating that similar
    content was muted, dismissed, ignored, or opted out.
    """

    signal_terms = [
        "mute",
        "muted",
        "dismiss",
        "dismissed",
        "ignore",
        "ignored",
        "opted out",
        "opt-out",
        "unsubscribe",
        "unsubscribed",
    ]

    for table_name in [
        "message_history",
        "message_events",
        "user_business_history",
    ]:

        table = history.get(table_name)

        if table is None or table.empty:
            continue

        # Search all textual columns.
        for column in table.columns:

            if table[column].dtype == "object":
                values = (
                    table[column]
                    .dropna()
                    .astype(str)
                    .str.lower()
                )

                for value in values:
                    if any(term in value for term in signal_terms):
                        return True

    return False


# ============================================================
# BUSINESS / SENDER CONTEXT
# ============================================================

def sender_is_known_business(data, row):
    """
    Check whether the message's business_id corresponds to a
    known business account.
    """

    business_id = clean_text(row.get("business_id", ""))

    if not business_id:
        return False

    businesses = data.get("business_accounts")

    if businesses is None or businesses.empty:
        return False

    if "business_id" not in businesses.columns:
        return False

    matches = businesses[
        businesses["business_id"].astype(str) == str(business_id)
    ]

    return not matches.empty


def user_has_business_history(data, row):
    """
    Check whether the user has an existing relationship with
    the business sending the message.
    """

    user_id = clean_text(row.get("user_id", ""))
    business_id = clean_text(row.get("business_id", ""))

    if not user_id or not business_id:
        return False

    history = data.get("user_business_history")

    if history is None or history.empty:
        return False

    if "user_id" not in history.columns:
        return False

    if "business_id" not in history.columns:
        return False

    matches = history[
        (history["user_id"].astype(str) == str(user_id))
        &
        (history["business_id"].astype(str) == str(business_id))
    ]

    return not matches.empty


# ============================================================
# ROUTING
# ============================================================

def route_message(row, data):
    """
    Main routing decision.

    Priority:

        1. Safety
        2. User history
        3. Urgency
        4. Event/business relevance
        5. General usefulness
    """

    text = clean_text(row.get("message_text", ""))
    normalized = normalize_text(text)

    user_id = clean_text(row.get("user_id", ""))

    # --------------------------------------------------------
    # EMPTY / UNREADABLE MESSAGE
    # --------------------------------------------------------

    if not normalized:
        return (
            "digest",
            "personal",
            "No readable message content indicates an immediate "
            "action or safety concern.",
            0.60,
            [],
        )

    # --------------------------------------------------------
    # RETRIEVE USER CONTEXT
    # --------------------------------------------------------

    history = get_user_history(data, user_id)

    evidence = find_history_evidence(history)

    # --------------------------------------------------------
    # SAFETY FIRST
    # --------------------------------------------------------

    injection = detect_prompt_injection(text)
    scam = detect_scam(text)

    if scam:

        if injection:
            reason = (
                "The message attempts to manipulate the routing agent "
                "and also requests sensitive verification information "
                "under account or payment pressure."
            )
        else:
            reason = (
                "The message requests sensitive verification information "
                "and uses account or payment pressure, indicating a "
                "likely scam."
            )

        return (
            "mute",
            "scam",
            reason,
            0.87,
            evidence[:3],
        )

    # --------------------------------------------------------
    # FORWARD / REPEATED UNWANTED CONTENT
    # --------------------------------------------------------

    unwanted_history = history_contains_unwanted_behavior(history)

    if unwanted_history:

        if detect_promotion(text):
            return (
                "mute",
                "promotion",
                "Similar promotional content was previously "
                "muted, dismissed, ignored, or opted out of by the user.",
                0.85,
                evidence[:3],
            )

        if detect_forward(text):
            return (
                "mute",
                "forward",
                "The content appears to be a forwarded message and "
                "previous user behavior indicates similar content "
                "was unwanted.",
                0.83,
                evidence[:3],
            )

    # --------------------------------------------------------
    # URGENT MESSAGE
    # --------------------------------------------------------

    if detect_urgent(text):

        # A trusted business with an existing relationship can
        # strengthen confidence in an operational update.
        if sender_is_known_business(data, row):
            message_type = "business_update"
        else:
            message_type = "urgent"

        return (
            "notify",
            message_type,
            "The message contains a time-sensitive request, deadline, "
            "or immediate action that may require the user's attention.",
            0.84,
            evidence[:3],
        )

    # --------------------------------------------------------
    # BUSINESS UPDATE
    # --------------------------------------------------------

    if sender_is_known_business(data, row):

        if user_has_business_history(data, row):

            return (
                "notify",
                "business_update",
                "The message comes from a known business with an "
                "existing user relationship, making the update "
                "relevant to the user.",
                0.84,
                evidence[:3],
            )

        return (
            "digest",
            "business_update",
            "The message appears to be a legitimate business update "
            "but does not require immediate attention.",
            0.78,
            evidence[:3],
        )

    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    if detect_event(text):

        return (
            "digest",
            "event",
            "The message contains useful event or schedule information "
            "but does not show a strong need for immediate interruption.",
            0.82,
            evidence[:3],
        )

    # --------------------------------------------------------
    # PROMOTION
    # --------------------------------------------------------

    if detect_promotion(text):

        return (
            "digest",
            "promotion",
            "The message contains promotional or offer-related "
            "information without a clear immediate action requirement.",
            0.75,
            evidence[:3],
        )

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if detect_greeting(text):

        return (
            "digest",
            "greeting",
            "The message is a non-urgent greeting and does not "
            "require immediate attention.",
            0.80,
            evidence[:3],
        )

    # --------------------------------------------------------
    # PERSONAL
    # --------------------------------------------------------

    if detect_personal(text):

        return (
            "digest",
            "personal",
            "The message is personal conversation without a clear "
            "immediate action requirement.",
            0.78,
            evidence[:3],
        )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return (
        "digest",
        "unknown",
        "The message does not show a clear reason for immediate "
        "interruption or muting.",
        0.60,
        evidence[:3],
    )


# ============================================================
# MESSAGE PROCESSING
# ============================================================

def process_message(row, data):
    """Process one message and return one output record."""

    message_id = clean_text(row.get("message_id", ""))

    (
        action,
        message_type,
        reason,
        confidence,
        evidence,
    ) = route_message(row, data)

    if not evidence:
        evidence_string = "none"
    else:
        evidence_string = ";".join(evidence)

    return {
        "message_id": message_id,
        "action": action,
        "message_type": message_type,
        "reason": reason,
        "confidence": round(
            max(0.0, min(1.0, safe_float(confidence))),
            2,
        ),
        "evidence_message_ids": evidence_string,
    }


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_output(output, messages):
    """Validate output before writing output.csv."""

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    if list(output.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "Incorrect output columns.\n"
            f"Expected: {OUTPUT_COLUMNS}\n"
            f"Got: {list(output.columns)}"
        )

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    if len(output) != len(messages):
        raise ValueError(
            f"Expected {len(messages)} output rows, "
            f"but generated {len(output)}."
        )

    # --------------------------------------------------------
    # Message IDs
    # --------------------------------------------------------

    if output["message_id"].duplicated().any():
        raise ValueError("Duplicate message_id found.")

    input_ids = set(messages["message_id"].astype(str))
    output_ids = set(output["message_id"].astype(str))

    if input_ids != output_ids:
        missing = input_ids - output_ids
        extra = output_ids - input_ids

        raise ValueError(
            f"Message ID mismatch. Missing={missing}, Extra={extra}"
        )

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    invalid_actions = (
        set(output["action"].astype(str)) - VALID_ACTIONS
    )

    if invalid_actions:
        raise ValueError(
            f"Invalid action values: {invalid_actions}"
        )

    # --------------------------------------------------------
    # Message types
    # --------------------------------------------------------

    invalid_types = (
        set(output["message_type"].astype(str))
        - VALID_MESSAGE_TYPES
    )

    if invalid_types:
        raise ValueError(
            f"Invalid message types: {invalid_types}"
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = pd.to_numeric(
        output["confidence"],
        errors="coerce",
    )

    if confidence.isna().any():
        raise ValueError("Confidence contains non-numeric values.")

    if not confidence.between(0, 1).all():
        raise ValueError(
            "Every confidence value must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    for column in OUTPUT_COLUMNS:
        if output[column].isna().any():
            raise ValueError(
                f"Missing values found in column: {column}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HACKERRANK ORCHESTRATE - MESSAGE NOTIFICATION ROUTER")
    print("=" * 60)

    print("\nLoading datasets...")

    data = load_data()

    if "messages" not in data:
        raise FileNotFoundError(
            "dataset/messages.csv was not found."
        )

    messages = data["messages"]

    print(f"Messages loaded: {len(messages)}")

    print("\nLoaded datasets:")

    for name, table in data.items():
        print(f"  {name}: {len(table)} rows")

    # --------------------------------------------------------
    # Process messages
    # --------------------------------------------------------

    print("\nProcessing messages...")

    results = []

    for _, row in messages.iterrows():

        result = process_message(
            row,
            data,
        )

        results.append(result)

    output = pd.DataFrame(
        results,
        columns=OUTPUT_COLUMNS,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print("Validating output...")

    validate_output(
        output,
        messages,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSUCCESS")
    print(f"Output written to: {OUTPUT_FILE}")
    print(f"Predictions generated: {len(output)}")

    print("\nAction distribution:")

    print(
        output["action"]
        .value_counts()
        .to_string()
    )

    print("\nMessage-type distribution:")

    print(
        output["message_type"]
        .value_counts()
        .to_string()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
