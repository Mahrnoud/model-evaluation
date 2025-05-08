"""
QA Evaluation Script with Enhanced Arabic Support

This script provides tools for evaluating question-answering models with
special support for Arabic language processing and evaluation metrics.

Usage:
    python main.py --model_path MODEL_PATH --tokenizer_path TOKENIZER_PATH --dataset_path DATASET_PATH

For full usage options:
    python main.py --help
"""

import os
import sys
import argparse
import time
import logging
import json


import pandas as pd

# First import the most basic modules with no dependencies
from utils.greedy_inference import load_model_and_tokenizer, greedy_decode
from evaluation.config import EvaluationConfig

# Now import the modules that depend on the basic ones

# Finally import the evaluation functions that use all the above
from evaluation.evaluator import evaluate_qa_model, evaluate_both_languages

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


def print_evaluation_summary(summary, lang):
    """Print a summary of evaluation results"""
    print("\nQA Evaluation Summary:")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Question language: {lang.upper()}")

    print("\nQuality Distribution:")
    for quality, count in summary['quality_distribution'].items():
        percentage = count / summary['total_samples'] * 100
        print(f"  {quality}: {count} ({percentage:.1f}%)")

    print("\nAverage Metrics:")
    for metric, value in summary['average_metrics'].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    print("\nCategory-wise Performance:")
    for category, metrics in summary['category_metrics'].items():
        print(f"  {category} (count: {metrics['count']}):")
        for metric, value in metrics.items():
            if metric != 'count' and isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")


def print_language_comparison(combined_summary):
    """Print a comparison of evaluation results across languages"""
    print("\nLanguage Comparison:")

    # Get results for each language
    language_results = combined_summary['language_results']
    languages = list(language_results.keys())

    # Print sample counts
    print("\nSample Counts:")
    for lang in languages:
        print(f"  {lang.upper()}: {language_results[lang]['total_samples']} samples")

    # Print quality distribution comparison
    print("\nQuality Distribution:")
    qualities = ["Excellent", "Good", "Partial", "Poor"]
    for quality in qualities:
        print(f"  {quality}:")
        for lang in languages:
            count = language_results[lang]['quality_distribution'].get(quality, 0)
            percentage = count / language_results[lang]['total_samples'] * 100
            print(f"    {lang.upper()}: {count} ({percentage:.1f}%)")

    # Print metrics comparison
    print("\nAverage Metrics:")
    metrics = ["bleu", "rouge1", "rouge2", "rougeL", "f1_score", "exact_match"]
    for metric in metrics:
        print(f"  {metric.upper()}:")
        values = []
        for lang in languages:
            value = language_results[lang]['average_metrics'].get(metric, 0)
            values.append(value)
            print(f"    {lang.upper()}: {value:.4f}")

        # Calculate difference if there are two languages
        if len(languages) == 2:
            diff = values[0] - values[1]
            print(f"    Difference: {diff:.4f}")

    # Print generation time comparison
    print("\nGeneration Time:")
    for lang in languages:
        time_value = language_results[lang]['average_metrics'].get('avg_generation_time', 0)
        print(f"  {lang.upper()}: {time_value:.4f} seconds")

    # If there are common categories, print category comparison
    common_categories = None
    if all('category_metrics' in language_results[lang] for lang in languages):
        categories1 = set(language_results[languages[0]]['category_metrics'].keys())
        if len(languages) > 1:
            categories2 = set(language_results[languages[1]]['category_metrics'].keys())
            common_categories = categories1.intersection(categories2)

            if common_categories:
                print("\nCategory-wise Performance Comparison (BLEU):")
                for category in common_categories:
                    print(f"  {category}:")
                    for lang in languages:
                        value = language_results[lang]['category_metrics'][category]['bleu']
                        print(f"    {lang.upper()}: {value:.4f}")


def compare_languages(model, tokenizer, device, config):
    """Perform a detailed comparison of QA performance between English and Arabic"""
    logger.info("Starting detailed language comparison analysis")

    # Create a config for each language
    en_config = EvaluationConfig(
        model_path=config.model_path,
        tokenizer_path=config.tokenizer_path,
        dataset_path=config.dataset_path,
        question_lang="en",
        sample_size=config.sample_size,
        output_dir=config.output_dir,
        max_length=config.max_length,
        model_config=config.model_config,
        prepare_human_eval=False,  # Skip human eval for comparison
        debug_mode=config.debug_mode
    )

    ar_config = EvaluationConfig(
        model_path=config.model_path,
        tokenizer_path=config.tokenizer_path,
        dataset_path=config.dataset_path,
        question_lang="ar",
        sample_size=config.sample_size,
        output_dir=config.output_dir,
        max_length=config.max_length,
        model_config=config.model_config,
        prepare_human_eval=False,
        debug_mode=config.debug_mode
    )

    # Evaluate both languages
    logger.info("Evaluating English QA performance")
    en_summary = evaluate_qa_model(model, tokenizer, device, en_config)

    logger.info("Evaluating Arabic QA performance")
    ar_summary = evaluate_qa_model(model, tokenizer, device, ar_config)

    # Load detailed results for deeper analysis
    en_results_file = os.path.join(config.output_dir, "detailed_results_en.json")
    ar_results_file = os.path.join(config.output_dir, "detailed_results_ar.json")

    with open(en_results_file, 'r', encoding='utf-8') as f:
        en_results = json.load(f)

    with open(ar_results_file, 'r', encoding='utf-8') as f:
        ar_results = json.load(f)

    # Create DataFrame for easier analysis
    en_df = pd.DataFrame(en_results)
    ar_df = pd.DataFrame(ar_results)

    # Perform comparative analysis
    comparison_results = {
        "language_results": {
            "en": en_summary,
            "ar": ar_summary
        },
        "comparative_analysis": {}
    }

    # 1. Performance by question length
    en_df['question_length'] = en_df['question'].apply(len)
    ar_df['question_length'] = ar_df['question'].apply(len)

    # Bin question lengths
    bins = [0, 50, 100, 150, 200, 300, 1000]
    bin_labels = ['0-50', '51-100', '101-150', '151-200', '201-300', '300+']

    en_df['length_bin'] = pd.cut(en_df['question_length'], bins=bins, labels=bin_labels)
    ar_df['length_bin'] = pd.cut(ar_df['question_length'], bins=bins, labels=bin_labels)

    # Calculate average scores by bin
    en_by_length = en_df.groupby('length_bin')['bleu'].mean().to_dict()
    ar_by_length = ar_df.groupby('length_bin')['bleu'].mean().to_dict()

    comparison_results['comparative_analysis']['performance_by_length'] = {
        "en": en_by_length,
        "ar": ar_by_length
    }

    # 2. Performance by answer length
    en_df['ref_length'] = en_df['reference_answer'].apply(len)
    ar_df['ref_length'] = ar_df['reference_answer'].apply(len)

    # Bin reference lengths
    ref_bins = [0, 100, 200, 300, 500, 1000, 10000]
    ref_bin_labels = ['0-100', '101-200', '201-300', '301-500', '501-1000', '1000+']

    en_df['ref_length_bin'] = pd.cut(en_df['ref_length'], bins=ref_bins, labels=ref_bin_labels)
    ar_df['ref_length_bin'] = pd.cut(ar_df['ref_length'], bins=ref_bins, labels=ref_bin_labels)

    # Calculate average scores by bin
    en_by_ref_length = en_df.groupby('ref_length_bin')['bleu'].mean().to_dict()
    ar_by_ref_length = ar_df.groupby('ref_length_bin')['bleu'].mean().to_dict()

    comparison_results['comparative_analysis']['performance_by_ref_length'] = {
        "en": en_by_ref_length,
        "ar": ar_by_ref_length
    }

    # 3. Length ratio analysis
    en_df['length_ratio'] = en_df['generated_answer'].apply(len) / en_df['reference_answer'].apply(len)
    ar_df['length_ratio'] = ar_df['generated_answer'].apply(len) / ar_df['reference_answer'].apply(len)

    comparison_results['comparative_analysis']['length_ratio'] = {
        "en": {
            "mean": en_df['length_ratio'].mean(),
            "median": en_df['length_ratio'].median(),
            "std": en_df['length_ratio'].std()
        },
        "ar": {
            "mean": ar_df['length_ratio'].mean(),
            "median": ar_df['length_ratio'].median(),
            "std": ar_df['length_ratio'].std()
        }
    }

    # 4. Common categories comparison
    if 'category' in en_df.columns and 'category' in ar_df.columns:
        # Find common categories
        en_categories = set(en_df['category'].unique())
        ar_categories = set(ar_df['category'].unique())
        common_categories = en_categories.intersection(ar_categories)

        category_comparison = {}
        for category in common_categories:
            en_category_df = en_df[en_df['category'] == category]
            ar_category_df = ar_df[ar_df['category'] == category]

            category_comparison[category] = {
                "en": {
                    "count": len(en_category_df),
                    "bleu": en_category_df['bleu'].mean(),
                    "f1_score": en_category_df['f1_score'].mean(),
                    "rougeL": en_category_df['rougeL'].mean()
                },
                "ar": {
                    "count": len(ar_category_df),
                    "bleu": ar_category_df['bleu'].mean(),
                    "f1_score": ar_category_df['f1_score'].mean(),
                    "rougeL": ar_category_df['rougeL'].mean()
                }
            }

        comparison_results['comparative_analysis']['category_comparison'] = category_comparison

    # 5. Error pattern analysis
    # Find samples with large score differences between metrics
    en_df['metric_diff'] = abs(en_df['bleu'] - en_df['f1_score'])
    ar_df['metric_diff'] = abs(ar_df['bleu'] - ar_df['f1_score'])

    # Get samples with high metric difference
    en_metric_diff = en_df[en_df['metric_diff'] > 0.3].sort_values('metric_diff', ascending=False).head(5).to_dict(
        'records')
    ar_metric_diff = ar_df[ar_df['metric_diff'] > 0.3].sort_values('metric_diff', ascending=False).head(5).to_dict(
        'records')

    comparison_results['comparative_analysis']['metric_disagreement_samples'] = {
        "en": en_metric_diff,
        "ar": ar_metric_diff
    }

    # 6. Generation time analysis
    comparison_results['comparative_analysis']['generation_time'] = {
        "en": {
            "mean": en_df['generation_time'].mean(),
            "median": en_df['generation_time'].median(),
            "std": en_df['generation_time'].std(),
            "min": en_df['generation_time'].min(),
            "max": en_df['generation_time'].max()
        },
        "ar": {
            "mean": ar_df['generation_time'].mean(),
            "median": ar_df['generation_time'].median(),
            "std": ar_df['generation_time'].std(),
            "min": ar_df['generation_time'].min(),
            "max": ar_df['generation_time'].max()
        }
    }

    # Save comparison results
    comparison_file = os.path.join(config.output_dir, "detailed_language_comparison.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Detailed language comparison complete. Results saved to {comparison_file}")

    return comparison_results


def main():
    """Main function to run the evaluation"""
    # Start timing
    start_time = time.time()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate QA model on a bilingual dataset with enhanced Arabic support')

    # Model and data paths
    parser.add_argument('--model_path', type=str, default="data/model_full.pth",
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default="miscovery/tokenizer",
                        help='Path to tokenizer')
    parser.add_argument('--dataset_path', type=str, default="data/train.csv",
                        help='Path to CSV or JSON dataset')

    # Evaluation settings
    parser.add_argument('--mode', type=str, default='both', choices=['single', 'both', 'compare'],
                        help='Evaluation mode: single language, both languages, or compare')
    parser.add_argument('--question_lang', type=str, default='en', choices=['en', 'ar'],
                        help='Question language to evaluate (for single mode)')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='qa_evaluation_results',
                        help='Output directory')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum answer generation length')
    parser.add_argument('--human_eval', action='store_true',
                        help='Prepare human evaluation sheet')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing for evaluation')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker processes for parallel evaluation')

    # Visualization and reporting
    parser.add_argument('--no_visualize', action='store_true',
                        help='Skip generating visualizations')
    parser.add_argument('--html_report', action='store_true',
                        help='Generate HTML report')

    # Testing and validation
    parser.add_argument('--validate', action='store_true',
                        help='Validate environment before running')
    parser.add_argument('--stress_test', action='store_true',
                        help='Run stress test on model')
    parser.add_argument('--run_tests', action='store_true',
                        help='Run test suite')

    # Optional model configuration parameters
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed-forward dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')

    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--error_analysis', action='store_true', help='Perform detailed error analysis')

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug or args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    # Run tests if requested
    if args.run_tests:
        logger.info("Running test suite")
        # Import testing module only when needed
        from testing import run_tests
        run_tests()
        return

    # Validate environment if requested
    if args.validate:
        logger.info("Validating evaluation environment")
        # Import validation module only when needed
        from testing.validation import validate_evaluation_environment
        validation_results = validate_evaluation_environment()

        # Print validation results
        print("\nEnvironment Validation Results:")
        print(f"- Python version: {validation_results['system']['python_version']}")
        print(f"- OS: {validation_results['system']['os']}")

        print("\nDependencies:")
        for package, info in validation_results['dependencies'].items():
            status = "✓" if info['installed'] else "✗"
            version = info.get('version', 'N/A')
            print(f"- {status} {package} ({version}): {info['purpose']}")

        print("\nGPU Availability:")
        if isinstance(validation_results['gpu'], dict):
            if validation_results['gpu']['available']:
                print(f"- GPU detected: {validation_results['gpu']['device_name']}")
            else:
                print("- No GPU detected. Evaluation will be slower on CPU.")
        else:
            print(f"- {validation_results['gpu']}")

        if validation_results['warnings']:
            print("\nWarnings:")
            for warning in validation_results['warnings']:
                print(f"- {warning}")

        if validation_results['critical_issues']:
            print("\nCritical Issues (must be resolved):")
            for issue in validation_results['critical_issues']:
                print(f"- {issue}")
            return

        print("\nEnvironment validation complete. Continuing with evaluation...\n")

    # Create model configuration
    model_config = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        # vocab_size and pad_token_id will be set based on tokenizer
    }

    # Create evaluation configuration
    config = EvaluationConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset_path,
        question_lang=args.question_lang,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        max_length=args.max_length,
        model_config=model_config,
        prepare_human_eval=args.human_eval,
        debug_mode=args.debug
    )

    # Load model and tokenizer
    logger.info("Loading model and tokenizer")
    try:
        model, tokenizer, device = load_model_and_tokenizer(
            args.model_path, args.tokenizer_path, model_config
        )
        logger.info(f"Model loaded successfully, running on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Run stress test if requested
    if args.stress_test:
        logger.info("Running stress test on model")
        # Import stress test module only when needed
        from testing.validation import run_stress_test
        stress_results = run_stress_test(model, tokenizer, device)

        # Print stress test results
        print("\nStress Test Results:")
        if 'error' in stress_results:
            print(f"Error during stress test: {stress_results['error']}")
        elif 'warning' in stress_results:
            print(f"Warning: {stress_results['warning']}")
        else:
            print("\nBatch Size Results:")
            for result in stress_results['batch_size_results']:
                if 'error' in result:
                    print(f"- Batch size {result['batch_size']}: Failed - {result['error']}")
                else:
                    print(
                        f"- Batch size {result['batch_size']}: {result['memory_mb']:.2f}MB, {result['time_seconds']:.2f}s, {result['time_per_item']:.2f}s per item")

            print(f"\nRecommended batch size: {stress_results['recommended_batch_size']}")

            print("\nMax Length Results:")
            for result in stress_results['max_length_results']:
                if 'error' in result:
                    print(f"- Max length {result['max_length']}: Failed - {result['error']}")
                else:
                    print(
                        f"- Max length {result['max_length']}: {result['memory_mb']:.2f}MB, {result['time_seconds']:.2f}s")

            print(f"\nRecommended max length: {stress_results['recommended_max_length']}")

    # Run evaluation
    if args.mode == 'single':
        summary = evaluate_qa_model(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config
        )

        # Print summary
        print_evaluation_summary(summary, config.question_lang)

    elif args.mode == 'compare':
        # Special comparison mode with more detailed analysis
        combined_summary = compare_languages(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config
        )

        # Print comparison
        print_language_comparison(combined_summary)

    else:  # args.mode == 'both'
        combined_summary = evaluate_both_languages(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config
        )

        # Print comparison
        print_language_comparison(combined_summary)

    # Print total execution time
    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    print(f"\nEvaluation completed in {execution_time:.2f} seconds")

    # Suggest next steps
    print("\nNext steps:")
    print(f"- Check detailed results in {config.output_dir}")
    if not args.no_visualize:
        print(f"- Review visualizations in {config.output_dir}")
    if args.html_report:
        if args.mode == 'single':
            report_path = os.path.join(config.output_dir, f"{config.question_lang}_evaluation_report.html")
        else:
            report_path = os.path.join(config.output_dir, "bilingual_comparison_report.html")
        print(f"- Open HTML report: {report_path}")
    if args.human_eval:
        human_eval_path = os.path.join(config.output_dir, f"human_evaluation_{config.question_lang}.xlsx")
        print(f"- Distribute human evaluation sheet: {human_eval_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback

        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        print("Check logs for details.")
        sys.exit(1)
