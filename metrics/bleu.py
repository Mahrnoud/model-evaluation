from difflib import SequenceMatcher
import logging

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# Import utils directly to avoid circular imports
from utils.text_processing import clean_text_for_comparison

# Configure logger
logger = logging.getLogger(__name__)


def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score between reference and hypothesis with better Arabic support

    Args:
        reference: Reference text (gold standard)
        hypothesis: Generated text to evaluate

    Returns:
        BLEU score
    """
    if not hypothesis or not reference:
        return 0.0

    # Clean and normalize texts first
    reference = clean_text_for_comparison(reference)
    hypothesis = clean_text_for_comparison(hypothesis)

    # Check if they're exactly equal after cleaning
    if reference == hypothesis:
        return 1.0

    # Check if the text is Arabic
    is_arabic = any('\u0600' <= c <= '\u06FF' for c in reference)

    # Decide on tokenization approach
    if is_arabic:
        # Import here to avoid circular dependency
        from language.arabic import normalize_arabic, improved_tokenize_arabic

        # First normalize
        reference = normalize_arabic(reference)
        hypothesis = normalize_arabic(hypothesis)

        # If they're equal after normalization, return 1.0
        if reference == hypothesis:
            return 1.0

        # Try word-level tokenization first
        reference_tokens = improved_tokenize_arabic(reference)
        hypothesis_tokens = improved_tokenize_arabic(hypothesis)

        # If tokenization failed or produced empty results, fall back to character-level
        if not reference_tokens or not hypothesis_tokens:
            # Character-level tokenization as fallback
            reference_tokens = list(reference)
            hypothesis_tokens = list(hypothesis)
    else:
        # For non-Arabic text, use word-level tokenization
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()

        # Remove stopwords if requested (for non-Arabic)
        try:
            from nltk.corpus import stopwords
            english_stopwords = set(stopwords.words('english'))
            reference_tokens = [token for token in reference_tokens if token not in english_stopwords]
            hypothesis_tokens = [token for token in hypothesis_tokens if token not in english_stopwords]
        except:
            # If NLTK stopwords aren't available, continue without removing stopwords
            pass

    # Use smoothing function to handle edge cases
    smoothie = SmoothingFunction().method1

    # Set weights for different n-grams (focus more on unigrams for Arabic)
    if is_arabic:
        weights = (0.7, 0.3, 0.0, 0.0)  # More weight on unigrams for Arabic
    else:
        weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for non-Arabic

    try:
        # Calculate BLEU score
        bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens,
                                   weights=weights, smoothing_function=smoothie)

        # Add boost for high character-level similarity
        if bleu_score < 0.9:  # Only if not already high
            char_sim = SequenceMatcher(None, reference, hypothesis).ratio()
            if char_sim > 0.8:
                # Boost BLEU score for high character similarity
                bleu_score = max(bleu_score, 0.7)

        return bleu_score
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {e}")
        return 0.0