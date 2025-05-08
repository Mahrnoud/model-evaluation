import re
from difflib import SequenceMatcher

# Import utils directly to avoid circular imports
from utils.text_processing import clean_text_for_comparison


def calculate_f1_word_match(reference, hypothesis, exclude_stopwords=True):
    """
    Calculate F1 score for word-level match, which works well for Arabic

    Args:
        reference: Reference text (gold standard)
        hypothesis: Generated text to evaluate
        exclude_stopwords: Whether to exclude stopwords from the calculation

    Returns:
        F1 score for word overlap
    """
    if not reference or not hypothesis:
        return 0.0

    # Clean and normalize texts
    reference = clean_text_for_comparison(reference)
    hypothesis = clean_text_for_comparison(hypothesis)

    # Check if they're exactly equal after cleaning
    if reference == hypothesis:
        return 1.0

    # Check if it's Arabic text
    is_arabic = any('\u0600' <= c <= '\u06FF' for c in reference)

    # For Arabic text, first normalize
    if is_arabic:
        # Import here to avoid circular dependency
        from language.arabic import normalize_arabic, improved_tokenize_arabic

        reference = normalize_arabic(reference)
        hypothesis = normalize_arabic(hypothesis)

        # If they're equal after normalization, return perfect score
        if reference == hypothesis:
            return 1.0

        # Check equality after removing all whitespace
        if re.sub(r'\s+', '', reference) == re.sub(r'\s+', '', hypothesis):
            return 0.95

        # Tokenize using improved function
        ref_tokens = improved_tokenize_arabic(reference)
        hyp_tokens = improved_tokenize_arabic(hypothesis)

        # Normalize Arabic words to handle spelling variations
        ref_tokens = [normalize_arabic(token) for token in ref_tokens]
        hyp_tokens = [normalize_arabic(token) for token in hyp_tokens]

        # Remove stopwords if requested
        if exclude_stopwords:
            # Import here to avoid circular dependency
            from language.arabic import get_arabic_stopwords
            stopwords = get_arabic_stopwords()
            ref_tokens = [token for token in ref_tokens if token not in stopwords]
            hyp_tokens = [token for token in hyp_tokens if token not in stopwords]
    else:
        # For non-Arabic text
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        # Remove stopwords if requested (for non-Arabic)
        if exclude_stopwords:
            try:
                from nltk.corpus import stopwords
                english_stopwords = set(stopwords.words('english'))
                ref_tokens = [token for token in ref_tokens if token not in english_stopwords]
                hyp_tokens = [token for token in hyp_tokens if token not in english_stopwords]
            except:
                pass

    # Remove empty tokens
    ref_tokens = [t for t in ref_tokens if t.strip()]
    hyp_tokens = [t for t in hyp_tokens if t.strip()]

    # Handle empty token lists
    if not ref_tokens or not hyp_tokens:
        # Check character similarity as fallback
        char_sim = SequenceMatcher(None, reference, hypothesis).ratio()
        return char_sim

    # Convert to multisets (allows duplicates)
    from collections import Counter
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)

    # Calculate intersection size
    intersection = sum((ref_counter & hyp_counter).values())

    # Calculate precision and recall
    precision = intersection / sum(hyp_counter.values()) if hyp_counter else 0
    recall = intersection / sum(ref_counter.values()) if ref_counter else 0

    # Calculate F1
    if precision + recall == 0:
        # Fall back to character-level similarity
        char_sim = SequenceMatcher(None, reference, hypothesis).ratio()
        if char_sim > 0.6:  # Only use if reasonably similar
            return char_sim
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)

    # For Arabic, consider character-level similarity to catch near-matches
    if is_arabic and f1 < 0.7:
        char_sim = SequenceMatcher(None, reference, hypothesis).ratio()
        if char_sim > 0.8:  # High character similarity
            return max(f1, 0.7)  # Boost score if character similarity is high

    return f1


def calculate_exact_match(reference, hypothesis):
    """
    Enhanced exact match that handles different types of whitespace and normalization

    Args:
        reference: Reference text
        hypothesis: Generated text

    Returns:
        Score between 0.0 and 1.0 indicating exact match quality
    """
    if not reference or not hypothesis:
        return 0.0

    # Clean texts for comparison
    clean_ref = clean_text_for_comparison(reference)
    clean_hyp = clean_text_for_comparison(hypothesis)

    # Direct match after cleaning
    if clean_ref == clean_hyp:
        return 1.0

    # Check if the text is Arabic
    is_arabic = any('\u0600' <= c <= '\u06FF' for c in reference)

    if is_arabic:
        # Import here to avoid circular dependency
        from language.arabic import normalize_arabic

        # Normalize Arabic texts
        norm_ref = normalize_arabic(reference)
        norm_hyp = normalize_arabic(hypothesis)

        # Match after normalization
        if norm_ref == norm_hyp:
            return 1.0

        # More lenient match - remove all whitespace
        if re.sub(r'\s+', '', norm_ref) == re.sub(r'\s+', '', norm_hyp):
            return 0.95

        # Even more lenient - compare only the actual Arabic characters
        ref_chars = ''.join(c for c in norm_ref if '\u0600' <= c <= '\u06FF')
        hyp_chars = ''.join(c for c in norm_hyp if '\u0600' <= c <= '\u06FF')

        if ref_chars == hyp_chars:
            return 0.9
    else:
        # For non-Arabic, try lowercase comparison
        if clean_ref.lower() == clean_hyp.lower():
            return 0.95

    # Calculate character-level similarity as fallback
    similarity = SequenceMatcher(None, clean_ref, clean_hyp).ratio()

    # Scale similarity to be more lenient
    if similarity > 0.9:
        return 0.9
    elif similarity > 0.8:
        return 0.8
    elif similarity > 0.7:
        return 0.7

    return 0.0
