import re


def clean_text_for_comparison(text):
    """
    Clean text for comparison, removing any potential invisible characters
    or unusual whitespace that might affect string comparison

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Convert to string
    text = str(text)

    # Remove any control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Replace various types of spaces with a standard space
    text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]', ' ', text)

    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
