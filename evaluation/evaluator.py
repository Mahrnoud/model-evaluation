import json
import os
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
from tqdm import tqdm
import logging

# Import directly from the module to avoid circular imports
from language.arabic import normalize_arabic
from metrics.bleu import calculate_bleu
from metrics.rouge import calculate_rouge
from metrics.exact_match import calculate_exact_match, calculate_f1_word_match
from utils.text_processing import clean_text_for_comparison
from utils.io import load_dataset

# Remove import from main.py
# from main import load_model_and_tokenizer, greedy_decode
from utils.greedy_inference import load_model_and_tokenizer, greedy_decode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("qa_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

# If visualization module is needed, import it directly
try:
    from visualization.reports import prepare_human_evaluation_sheet
    from visualization.plots import create_visualizations, create_comparative_visualization
except ImportError:
    logger.warning("Visualization modules not available, visualizations will be skipped")


    # Define placeholder functions if imports fail
    def prepare_human_evaluation_sheet(results, output_path, sample_size=50):
        logger.warning("Human evaluation sheet preparation skipped due to missing modules")
        return None


    def create_visualizations(results, evaluation_metrics, category_metrics, output_dir, lang_prefix):
        logger.warning("Visualization creation skipped due to missing modules")
        return None


    def create_comparative_visualization(results, output_dir):
        logger.warning("Comparative visualization skipped due to missing modules")
        return None


def evaluate_qa_model(
        model,
        tokenizer,
        device,
        config
):
    """
    Evaluate QA model performance on a dataset with enhanced Arabic support

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        device: Device to run on (cuda or cpu)
        config: Configuration object with evaluation settings

    Returns:
        Dictionary with evaluation summary
    """
    # Get parameters from config
    model_path = config.model_path
    tokenizer_path = config.tokenizer_path
    dataset_path = config.dataset_path
    question_lang = config.question_lang
    sample_size = config.sample_size
    output_dir = config.output_dir
    max_length = config.max_length
    prepare_human_eval = config.prepare_human_eval
    debug_mode = config.debug_mode

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    questions, answers, categories, sub_categories, df = load_dataset(dataset_path, sample_size, question_lang)

    # Prepare results storage
    results = []
    detailed_analyses = []  # For storing detailed analysis in debug mode
    discrepancy_samples = []  # For storing samples with metric discrepancies

    evaluation_metrics = {
        'bleu': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'f1_score': [],
        'exact_match': [],
        'generation_time': [],
    }

    category_metrics = defaultdict(lambda: defaultdict(list))

    # Evaluate model on each sample
    logger.info(f"Starting QA evaluation on {len(df)} samples in {question_lang} language")
    for idx, (question, reference_answer, category, sub_category) in enumerate(
            tqdm(zip(questions, answers, categories, sub_categories), total=len(df))
    ):
        # Generate answer
        start_time = time.time()
        try:
            generated_answer = greedy_decode(
                model=model,
                tokenizer=tokenizer,
                prompt=question,
                max_length=max_length,
                device=device
            )
        except Exception as e:
            logger.error(f"Error generating answer for question {idx}: {e}")
            generated_answer = ""

        generation_time = time.time() - start_time

        # Perform detailed analysis if in debug mode
        if debug_mode:
            analysis = analyze_sample(reference_answer, generated_answer, idx, question_lang)
            detailed_analyses.append(analysis)

            # Check for metric discrepancies
            if analysis.get('has_metric_discrepancy', False):
                discrepancy_samples.append(analysis)

        # Calculate standard metrics
        bleu_score = calculate_bleu(reference_answer, generated_answer)
        rouge_scores = calculate_rouge(reference_answer, generated_answer)
        f1_score = calculate_f1_word_match(reference_answer, generated_answer)
        exact_match = calculate_exact_match(reference_answer, generated_answer)

        # For Arabic, normalize before storing
        if question_lang == 'ar':
            normalized_reference = normalize_arabic(reference_answer)
            normalized_generated = normalize_arabic(generated_answer)
        else:
            normalized_reference = reference_answer
            normalized_generated = generated_answer

        # Determine quality level
        if exact_match > 0.9:
            quality = "Excellent"
        elif f1_score > 0.7 or bleu_score > 0.7:
            quality = "Good"
        elif f1_score > 0.4 or bleu_score > 0.4:
            quality = "Partial"
        else:
            quality = "Poor"

        # Store results
        sample_result = {
            'id': idx,
            'question': question,
            'reference_answer': reference_answer,
            'generated_answer': generated_answer,
            'normalized_reference': normalized_reference,
            'normalized_generated': normalized_generated,
            'bleu': bleu_score,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'f1_score': f1_score,
            'exact_match': exact_match,
            'generation_time': generation_time,
            'category': category,
            'sub_category': sub_category,
            'quality': quality
        }

        results.append(sample_result)

        # Update metrics
        evaluation_metrics['bleu'].append(bleu_score)
        evaluation_metrics['rouge1'].append(rouge_scores['rouge1'])
        evaluation_metrics['rouge2'].append(rouge_scores['rouge2'])
        evaluation_metrics['rougeL'].append(rouge_scores['rougeL'])
        evaluation_metrics['f1_score'].append(f1_score)
        evaluation_metrics['exact_match'].append(exact_match)
        evaluation_metrics['generation_time'].append(generation_time)

        # Update category metrics
        category_metrics[category]['bleu'].append(bleu_score)
        category_metrics[category]['rouge1'].append(rouge_scores['rouge1'])
        category_metrics[category]['rouge2'].append(rouge_scores['rouge2'])
        category_metrics[category]['rougeL'].append(rouge_scores['rougeL'])
        category_metrics[category]['f1_score'].append(f1_score)
        category_metrics[category]['exact_match'].append(exact_match)

    # Calculate average metrics
    avg_metrics = {
        metric: np.mean(values) if values else 0.0
        for metric, values in evaluation_metrics.items()
        if metric != 'generation_time'
    }
    avg_metrics['avg_generation_time'] = np.mean(evaluation_metrics['generation_time'])

    # Calculate category-wise metrics
    category_avg_metrics = {}
    for category, metrics in category_metrics.items():
        category_avg_metrics[category] = {
            metric: np.mean(values) if values else 0.0
            for metric, values in metrics.items()
        }
        category_avg_metrics[category]['count'] = len(metrics['bleu'])

    # For Arabic, perform error analysis
    if question_lang == 'ar':
        error_patterns = analyze_errors(results, output_dir, lang=question_lang)

        # Prepare human evaluation sheet if requested
        if prepare_human_eval:
            human_eval_path = os.path.join(output_dir, f'human_evaluation_{question_lang}.xlsx')
            prepare_human_evaluation_sheet(results, human_eval_path)

    # Save detailed analyses if in debug mode
    if debug_mode and detailed_analyses:
        analyses_file = os.path.join(output_dir, f'detailed_analyses_{question_lang}.json')
        with open(analyses_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_analyses, f, ensure_ascii=False, indent=2)

        # Save discrepancy samples separately
        if discrepancy_samples:
            discrepancy_file = os.path.join(output_dir, f'metric_discrepancies_{question_lang}.json')
            with open(discrepancy_file, 'w', encoding='utf-8') as f:
                json.dump(discrepancy_samples, f, ensure_ascii=False, indent=2)

    # Create summary
    summary = {
        'total_samples': len(df),
        'language': question_lang,
        'average_metrics': avg_metrics,
        'category_metrics': category_avg_metrics,
        'quality_distribution': {
            'Excellent': len([r for r in results if r['quality'] == 'Excellent']),
            'Good': len([r for r in results if r['quality'] == 'Good']),
            'Partial': len([r for r in results if r['quality'] == 'Partial']),
            'Poor': len([r for r in results if r['quality'] == 'Poor'])
        }
    }

    # Save detailed results
    results_file = os.path.join(output_dir, f'detailed_results_{question_lang}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save summary
    summary_file = os.path.join(output_dir, f'summary_{question_lang}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Generate visualizations if visualization module is available
    try:
        create_visualizations(results, evaluation_metrics, category_avg_metrics, output_dir, question_lang)
    except Exception as e:
        logger.warning(f"Failed to create visualizations: {e}")

    logger.info(f"QA evaluation complete. Results saved to {output_dir}")

    return summary


def analyze_sample(reference, generated, sample_id, question_lang):
    """
    Perform detailed analysis on a single sample to debug metric issues

    Args:
        reference: Reference text
        generated: Generated text
        sample_id: Sample ID for logging
        question_lang: Language of the sample

    Returns:
        Dictionary with detailed analysis
    """
    # First, debug the text comparison to identify any issues
    debug_text_comparison(reference, generated, sample_id)

    # Clean and normalize texts
    clean_ref = clean_text_for_comparison(reference)
    clean_gen = clean_text_for_comparison(generated)

    # Calculate raw metrics without any normalization
    raw_metrics = {
        'exact_match_raw': 1.0 if clean_ref == clean_gen else 0.0,
        'char_similarity_raw': SequenceMatcher(None, clean_ref, clean_gen).ratio()
    }

    # Calculate metrics with normalization
    if question_lang == 'ar':
        norm_ref = normalize_arabic(reference)
        norm_gen = normalize_arabic(generated)

        norm_metrics = {
            'exact_match_norm': 1.0 if norm_ref == norm_gen else 0.0,
            'char_similarity_norm': SequenceMatcher(None, norm_ref, norm_gen).ratio()
        }

        # Check equality after removing all whitespace
        no_space_ref = re.sub(r'\s+', '', norm_ref)
        no_space_gen = re.sub(r'\s+', '', norm_gen)
        norm_metrics['exact_match_no_space'] = 1.0 if no_space_ref == no_space_gen else 0.0
    else:
        # For non-Arabic text
        norm_ref = clean_ref.lower()
        norm_gen = clean_gen.lower()

        norm_metrics = {
            'exact_match_norm': 1.0 if norm_ref == norm_gen else 0.0,
            'char_similarity_norm': SequenceMatcher(None, norm_ref, norm_gen).ratio()
        }

        # Check equality after removing all whitespace
        no_space_ref = re.sub(r'\s+', '', norm_ref)
        no_space_gen = re.sub(r'\s+', '', norm_gen)
        norm_metrics['exact_match_no_space'] = 1.0 if no_space_ref == no_space_gen else 0.0

    # Calculate all standard metrics
    bleu = calculate_bleu(reference, generated)
    rouge_scores = calculate_rouge(reference, generated)
    f1 = calculate_f1_word_match(reference, generated)
    exact_match = calculate_exact_match(reference, generated)

    # Combine all analysis
    analysis = {
        'sample_id': sample_id,
        'reference': reference,
        'generated': generated,
        'clean_reference': clean_ref,
        'clean_generated': clean_gen,
        'normalized_reference': norm_ref,
        'normalized_generated': norm_gen,
        'raw_metrics': raw_metrics,
        'norm_metrics': norm_metrics,
        'bleu': bleu,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'f1_score': f1,
        'exact_match': exact_match
    }

    # Determine if there's a significant discrepancy
    has_discrepancy = (
            (norm_metrics['exact_match_norm'] == 1.0 or norm_metrics['exact_match_no_space'] == 1.0)
            and (bleu < 0.9 or rouge_scores['rougeL'] < 0.9 or f1 < 0.9)
    )

    analysis['has_metric_discrepancy'] = has_discrepancy

    return analysis


def debug_text_comparison(reference, generated, id_num):
    """
    Debug helper to identify issues with text comparison

    Args:
        reference: Reference text
        generated: Generated text
        id_num: ID of the sample for logging
    """
    logger.debug(f"=== Debugging item {id_num} ===")
    logger.debug(f"Reference (len={len(reference)}): {repr(reference)}")
    logger.debug(f"Generated (len={len(generated)}): {repr(generated)}")

    # Check character by character
    if len(reference) == len(generated):
        mismatch_positions = []
        for i, (r, g) in enumerate(zip(reference, generated)):
            if r != g:
                mismatch_positions.append(f"Position {i}: '{r}' (\\u{ord(r):04x}) vs '{g}' (\\u{ord(g):04x})")

        if mismatch_positions:
            logger.debug("Character mismatches:")
            for pos in mismatch_positions[:10]:  # Limit to first 10 mismatches
                logger.debug(f"  {pos}")
            if len(mismatch_positions) > 10:
                logger.debug(f"  ... and {len(mismatch_positions) - 10} more mismatches")
        else:
            logger.debug("No character-by-character mismatches, but strings compare as unequal.")
            logger.debug("This suggests possible encoding or invisible character issues.")
    else:
        logger.debug(f"String lengths differ: reference={len(reference)}, generated={len(generated)}")

    # Check normalized versions
    norm_ref = normalize_arabic(reference)
    norm_gen = normalize_arabic(generated)
    logger.debug(f"Normalized reference (len={len(norm_ref)}): {repr(norm_ref)}")
    logger.debug(f"Normalized generated (len={len(norm_gen)}): {repr(norm_gen)}")
    logger.debug(f"Normalized strings equal? {norm_ref == norm_gen}")

    # Check after removing all whitespace
    no_space_ref = re.sub(r'\s+', '', norm_ref)
    no_space_gen = re.sub(r'\s+', '', norm_gen)
    logger.debug(f"No-space reference (len={len(no_space_ref)}): {repr(no_space_ref)}")
    logger.debug(f"No-space generated (len={len(no_space_gen)}): {repr(no_space_gen)}")
    logger.debug(f"No-space strings equal? {no_space_ref == no_space_gen}")


def analyze_errors(results, output_dir, lang='ar'):
    """
    Analyze common error patterns in the evaluation results

    Args:
        results: List of evaluation results
        output_dir: Directory to save analysis results
        lang: Language to analyze ('ar' for Arabic)

    Returns:
        Dictionary with error patterns and counts
    """
    error_patterns = defaultdict(int)
    error_examples = defaultdict(list)
    max_examples = 3  # Maximum number of examples to store per error pattern

    for result in results:
        ref = result.get('reference_answer', '')
        gen = result.get('generated_answer', '')

        if not ref or not gen:
            continue

        # Skip if exact match or very high similarity
        exact_match_score = calculate_exact_match(ref, gen)
        if exact_match_score > 0.9:
            continue

        # Check for common error types
        if len(gen) < len(ref) * 0.5:
            error_patterns['too_short'] += 1
            if len(error_examples['too_short']) < max_examples:
                error_examples['too_short'].append({
                    'question': result.get('question', ''),
                    'reference': ref,
                    'generated': gen
                })
        elif len(gen) > len(ref) * 1.5:
            error_patterns['too_long'] += 1
            if len(error_examples['too_long']) < max_examples:
                error_examples['too_long'].append({
                    'question': result.get('question', ''),
                    'reference': ref,
                    'generated': gen
                })

        # Check for missing dates, numbers, or proper nouns
        ref_dates = re.findall(r'\b\d{4}\b', ref)
        gen_dates = re.findall(r'\b\d{4}\b', gen)
        if ref_dates and not any(d in gen_dates for d in ref_dates):
            error_patterns['missing_dates'] += 1
            if len(error_examples['missing_dates']) < max_examples:
                error_examples['missing_dates'].append({
                    'question': result.get('question', ''),
                    'reference': ref,
                    'generated': gen,
                    'missing_date': ref_dates[0]
                })

        # Check for missing names (basic approach - look for capitalized words in ASCII text)
        # and words with Al- prefix in Arabic
        ref_names_ar = re.findall(r'\b(ال[\u0600-\u06FF]+)\b', ref)
        gen_names_ar = re.findall(r'\b(ال[\u0600-\u06FF]+)\b', gen)
        if ref_names_ar and not any(name in gen_names_ar for name in ref_names_ar):
            error_patterns['missing_names'] += 1
            if len(error_examples['missing_names']) < max_examples:
                error_examples['missing_names'].append({
                    'question': result.get('question', ''),
                    'reference': ref,
                    'generated': gen,
                    'missing_name': ref_names_ar[0]
                })

        # Check for wrong structure - different sentence count
        ref_sentences = len(re.split(r'[.!?؟،]', ref))
        gen_sentences = len(re.split(r'[.!?؟،]', gen))
        if abs(ref_sentences - gen_sentences) > 1:
            error_patterns['wrong_structure'] += 1
            if len(error_examples['wrong_structure']) < max_examples:
                error_examples['wrong_structure'].append({
                    'question': result.get('question', ''),
                    'reference': ref,
                    'generated': gen,
                    'ref_sentences': ref_sentences,
                    'gen_sentences': gen_sentences
                })

    # Save error analysis
    error_file = os.path.join(output_dir, f'{lang}_error_analysis.json')
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump({
            'error_counts': dict(error_patterns),
            'error_examples': error_examples
        }, f, ensure_ascii=False, indent=2)

    return error_patterns


def evaluate_both_languages(
        model,
        tokenizer,
        device,
        config
):
    """
    Evaluate QA model on both English and Arabic questions with enhanced metrics

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        device: Device to run on (cuda or cpu)
        config: Configuration object with evaluation settings

    Returns:
        Dictionary with combined evaluation summary
    """
    languages = ['en', 'ar']
    results = {}

    for lang in languages:
        print(f"\nEvaluating QA performance on {lang} questions:")

        # Create a language-specific config by copying and modifying the original
        lang_config = type(config)(
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            dataset_path=config.dataset_path,
            question_lang=lang,
            sample_size=config.sample_size,
            output_dir=config.output_dir,
            max_length=config.max_length,
            model_config=config.model_config,
            prepare_human_eval=config.prepare_human_eval,
            debug_mode=config.debug_mode
        )

        try:
            # Run evaluation for this language
            summary = evaluate_qa_model(
                model=model,
                tokenizer=tokenizer,
                device=device,
                config=lang_config
            )

            results[lang] = summary

            # Print summary
            print(f"  Language: {summary['language']}")
            print(f"  Total samples: {summary['total_samples']}")
            print("  Quality Distribution:")
            for quality, count in summary['quality_distribution'].items():
                percentage = count / summary['total_samples'] * 100
                print(f"    {quality}: {count} ({percentage:.1f}%)")
            print("  Average Metrics:")
            for metric, value in summary['average_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")

        except Exception as e:
            logger.error(f"Error evaluating {lang} questions: {e}")
            print(f"  Error: {e}")

    # Create a combined summary
    combined_summary = {
        'language_results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save combined summary
    combined_file = os.path.join(config.output_dir, 'combined_qa_summary.json')
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_summary, f, ensure_ascii=False, indent=2)

    # Create comparative visualization if visualization module is available
    try:
        create_comparative_visualization(results, config.output_dir)
    except Exception as e:
        logger.warning(f"Failed to create comparative visualization: {e}")

    return combined_summary
