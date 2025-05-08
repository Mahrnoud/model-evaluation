from language import normalize_arabic, improved_tokenize_arabic
from metrics import calculate_bleu, calculate_rouge, calculate_f1_word_match, calculate_exact_match
from utils import clean_text_for_comparison


def debug_metrics_calculation(reference, hypothesis):
    """
    Debug utility to trace through metrics calculation

    Args:
        reference: Reference text
        hypothesis: Generated text

    Returns:
        Dictionary with detailed metrics calculation info
    """
    debug_info = {
        'inputs': {
            'reference': reference,
            'hypothesis': hypothesis,
        },
        'text_processing': {
            'reference_cleaned': clean_text_for_comparison(reference),
            'hypothesis_cleaned': clean_text_for_comparison(hypothesis),
        }
    }

    # Check if it's Arabic
    is_arabic = any('\u0600' <= c <= '\u06FF' for c in reference)
    debug_info['is_arabic'] = is_arabic

    if is_arabic:
        debug_info['arabic_processing'] = {
            'reference_normalized': normalize_arabic(reference),
            'hypothesis_normalized': normalize_arabic(hypothesis),
        }

        # Add tokenization results
        ref_tokens = improved_tokenize_arabic(normalize_arabic(reference))
        hyp_tokens = improved_tokenize_arabic(normalize_arabic(hypothesis))
        debug_info['tokenization'] = {
            'reference_tokens': ref_tokens,
            'hypothesis_tokens': hyp_tokens,
        }
    else:
        # Add tokenization for non-Arabic
        debug_info['tokenization'] = {
            'reference_tokens': reference.lower().split(),
            'hypothesis_tokens': hypothesis.lower().split(),
        }

    # Calculate and trace all metrics
    try:
        bleu_score = calculate_bleu(reference, hypothesis)
        debug_info['bleu'] = {
            'score': bleu_score,
        }
    except Exception as e:
        debug_info['bleu'] = {
            'error': str(e),
        }

    try:
        rouge_scores = calculate_rouge(reference, hypothesis)
        debug_info['rouge'] = rouge_scores
    except Exception as e:
        debug_info['rouge'] = {
            'error': str(e),
        }

    try:
        f1_score = calculate_f1_word_match(reference, hypothesis)
        debug_info['f1'] = {
            'score': f1_score,
        }
    except Exception as e:
        debug_info['f1'] = {
            'error': str(e),
        }

    try:
        exact_match = calculate_exact_match(reference, hypothesis)
        debug_info['exact_match'] = {
            'score': exact_match,
        }
    except Exception as e:
        debug_info['exact_match'] = {
            'error': str(e),
        }

    return debug_info