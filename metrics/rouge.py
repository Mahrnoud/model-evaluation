# Set up logging
import logging
from difflib import SequenceMatcher

from rouge_score import rouge_scorer

# Import utils directly to avoid circular imports
from utils.text_processing import clean_text_for_comparison

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("qa_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)


def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores between reference and hypothesis with better Arabic support

    Args:
        reference: Reference text (gold standard)
        hypothesis: Generated text to evaluate

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    if not hypothesis or not reference:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    # Clean and normalize texts
    reference = clean_text_for_comparison(reference)
    hypothesis = clean_text_for_comparison(hypothesis)

    # Check if they're exactly equal after cleaning
    if reference == hypothesis:
        return {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0}

    try:
        # For Arabic text, first normalize
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in reference)

        if is_arabic:
            # Import here to avoid circular dependency
            from language.arabic import normalize_arabic, improved_tokenize_arabic

            # Normalize Arabic texts
            reference_normalized = normalize_arabic(reference)
            hypothesis_normalized = normalize_arabic(hypothesis)

            # If they're equal after normalization, return perfect scores
            if reference_normalized == hypothesis_normalized:
                return {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0}

            # Tokenize Arabic text
            reference_tokens = improved_tokenize_arabic(reference_normalized)
            hypothesis_tokens = improved_tokenize_arabic(hypothesis_normalized)

            # Join tokens with spaces for ROUGE scoring
            reference_processed = ' '.join(reference_tokens)
            hypothesis_processed = ' '.join(hypothesis_tokens)
        else:
            # For non-Arabic, just use the text as is
            reference_processed = reference
            hypothesis_processed = hypothesis

        # Initialize Rouge scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # Calculate scores with processed text
        scores = scorer.score(reference_processed, hypothesis_processed)

        # Extract F1 scores
        rouge_scores = {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

        # For Arabic, check character-level similarity as well
        if is_arabic and all(s < 0.9 for s in rouge_scores.values()):
            char_sim = SequenceMatcher(None, reference_normalized, hypothesis_normalized).ratio()
            if char_sim > 0.8:
                # Boost ROUGE scores for high character similarity
                rouge_scores = {k: max(v, 0.7) for k, v in rouge_scores.items()}

        return rouge_scores
    except Exception as e:
        logger.error(f"Error calculating ROUGE scores: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
