import re


def clean_english_text(text):
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
