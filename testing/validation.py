"""
Validation module for QA evaluation framework.

This module provides tools for validating the environment,
checking dependencies, and performing stress testing.
"""

import os
import sys
import time
import importlib
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def validate_evaluation_environment() -> Dict[str, Any]:
    """
    Validate the evaluation environment and dependencies

    Returns:
        Dictionary with validation results
    """
    validation = {
        'system': {
            'os': os.name,
            'python_version': sys.version,
        },
        'dependencies': {},
        'gpu': None,
        'warnings': [],
        'critical_issues': [],
    }

    # Check required dependencies
    dependencies = {
        'numpy': 'For numerical operations',
        'pandas': 'For data handling',
        'matplotlib': 'For visualization',
        'nltk': 'For BLEU scoring',
        'rouge_score': 'For ROUGE scoring',
        'tqdm': 'For progress bars',
        'torch': 'For model loading and inference',
        'transformers': 'For embedding-based similarity (optional)',
    }

    for package, purpose in dependencies.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            validation['dependencies'][package] = {
                'installed': True,
                'version': version,
                'purpose': purpose,
            }
        except ImportError:
            validation['dependencies'][package] = {
                'installed': False,
                'purpose': purpose,
            }
            if package in ['numpy', 'pandas', 'nltk', 'rouge_score', 'torch']:
                validation['critical_issues'].append(f"Missing critical dependency: {package}")
            else:
                validation['warnings'].append(f"Missing optional dependency: {package}")

    # Check for Arabic-specific packages
    arabic_dependencies = {
        'pyarabic': 'For Arabic tokenization',
        'camel_tools': 'For Arabic NLP',
        'farasapy': 'For advanced Arabic tokenization',
    }

    for package, purpose in arabic_dependencies.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            validation['dependencies'][package] = {
                'installed': True,
                'version': version,
                'purpose': purpose,
            }
        except ImportError:
            validation['dependencies'][package] = {
                'installed': False,
                'purpose': purpose,
            }
            validation['warnings'].append(f"Missing Arabic NLP dependency: {package}")

    # Check NLTK data
    try:
        from nltk.data import find
        try:
            find('tokenizers/punkt')
            validation['nltk_data'] = {
                'punkt': True,
            }
        except LookupError:
            validation['nltk_data'] = {
                'punkt': False,
            }
            validation['warnings'].append("NLTK punkt tokenizer not found. Run nltk.download('punkt')")

        try:
            find('corpora/stopwords')
            validation['nltk_data']['stopwords'] = True
        except LookupError:
            validation['nltk_data']['stopwords'] = False
            validation['warnings'].append("NLTK stopwords not found. Run nltk.download('stopwords')")
    except Exception as e:
        validation['nltk_data'] = f'Unable to check NLTK data: {e}'

    # Check GPU availability
    try:
        import torch
        validation['gpu'] = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

        if not torch.cuda.is_available():
            validation['warnings'].append("No GPU detected. Evaluation will be slower on CPU.")
    except Exception as e:
        validation['gpu'] = f'Unable to check GPU: {e}'
        validation['warnings'].append(f"Unable to check GPU availability: {e}")

    # Check disk space for output
    try:
        import shutil
        disk_usage = shutil.disk_usage('.')
        free_space_gb = disk_usage.free / (1024 ** 3)
        validation['disk_space'] = {
            'free_gb': free_space_gb,
            'sufficient': free_space_gb > 1.0,  # Check if more than 1GB available
        }

        if free_space_gb < 1.0:
            validation['warnings'].append(
                f"Low disk space: {free_space_gb:.2f}GB free. Evaluation results may not be saved properly.")
    except Exception as e:
        validation['disk_space'] = f'Unable to check disk space: {e}'

    # Check for greedy_inference module
    try:
        import greedy_inference
        validation['custom_modules'] = {
            'greedy_inference': True
        }
    except ImportError:
        validation['custom_modules'] = {
            'greedy_inference': False
        }
        validation['critical_issues'].append("Missing greedy_inference module, which is required for model inference")

    return validation


def run_stress_test(model, tokenizer, device, batch_sizes=None, seq_lengths=None) -> Dict[str, Any]:
    """
    Perform stress testing on the model to determine optimal batch size and memory usage

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        device: Device to run on (cuda or cpu)
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8, 16])
        seq_lengths: List of sequence lengths to test (default: [128, 256, 512, 1024])

    Returns:
        Dictionary with stress test results
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]

    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024]

    results = {
        'batch_size_results': [],
        'recommended_batch_size': 1,
        'max_length_results': [],
        'recommended_max_length': 128,
    }

    # Import required modules
    try:
        import torch
        from greedy_inference import greedy_decode
    except ImportError as e:
        results['error'] = f"Required module not found: {e}"
        return results

    # Skip if not using GPU
    if 'cuda' not in str(device):
        results['warning'] = "Stress test skipped: no GPU detected"
        return results

    # Test batch sizes
    try:
        default_input = "This is a test input to measure performance."
        arabic_input = "هذا مدخل اختبار لقياس الأداء."

        for batch_size in batch_sizes:
            try:
                # Clear cache
                torch.cuda.empty_cache()

                # Create batch of inputs (mix of English and Arabic)
                inputs = [default_input] * (batch_size // 2) + [arabic_input] * (batch_size - batch_size // 2)

                # Measure time and memory
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated(device)

                # Tokenize and generate
                for input_text in inputs:
                    _ = greedy_decode(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=input_text,
                        max_length=128,
                        device=device
                    )

                end_time = time.time()
                end_memory = torch.cuda.memory_allocated(device)

                memory_used = (end_memory - start_memory) / (1024 ** 2)  # MB
                time_taken = end_time - start_time

                results['batch_size_results'].append({
                    'batch_size': batch_size,
                    'memory_mb': memory_used,
                    'time_seconds': time_taken,
                    'time_per_item': time_taken / batch_size,
                })

                # Check if memory usage is within reasonable limits (80% of available)
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if end_memory < 0.8 * total_memory:
                    results['recommended_batch_size'] = batch_size

            except RuntimeError as e:
                # Likely out of memory error
                results['batch_size_results'].append({
                    'batch_size': batch_size,
                    'error': str(e),
                    'status': 'failed',
                })
                break

        # Test sequence lengths
        for length in seq_lengths:
            try:
                # Clear cache
                torch.cuda.empty_cache()

                # Measure time and memory for English
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated(device)

                _ = greedy_decode(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=default_input,
                    max_length=length,
                    device=device
                )

                # Also test with Arabic
                _ = greedy_decode(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=arabic_input,
                    max_length=length,
                    device=device
                )

                end_time = time.time()
                end_memory = torch.cuda.memory_allocated(device)

                memory_used = (end_memory - start_memory) / (1024 ** 2)  # MB
                time_taken = end_time - start_time

                results['max_length_results'].append({
                    'max_length': length,
                    'memory_mb': memory_used,
                    'time_seconds': time_taken,
                })

                # Check if memory usage is within reasonable limits
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if end_memory < 0.8 * total_memory:
                    results['recommended_max_length'] = length

            except RuntimeError as e:
                # Likely out of memory error
                results['max_length_results'].append({
                    'max_length': length,
                    'error': str(e),
                    'status': 'failed',
                })
                break

    except Exception as e:
        results['error'] = str(e)
        logger.error(f"Error during stress test: {e}")

    return results


def validate_output_quality(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate the quality of generated answers for consistency

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with validation results
    """
    validation = {
        'warnings': [],
        'statistics': {},
    }

    if not results:
        validation['warnings'].append("No results to validate")
        return validation

    # Check for empty generations
    empty_generations = [r for r in results if not r.get('generated_answer')]
    if empty_generations:
        validation['warnings'].append(f"Found {len(empty_generations)} empty generated answers")

    # Check for very short generations
    short_generations = [r for r in results if r.get('generated_answer') and len(r['generated_answer']) < 10]
    if short_generations:
        validation['warnings'].append(f"Found {len(short_generations)} very short generated answers (< 10 chars)")

    # Check for very long generations
    long_generations = [r for r in results if r.get('generated_answer') and len(r['generated_answer']) > 1000]
    if long_generations:
        validation['warnings'].append(f"Found {len(long_generations)} very long generated answers (> 1000 chars)")

    # Check for abnormal length ratios
    abnormal_ratios = []
    for r in results:
        if r.get('reference_answer') and r.get('generated_answer'):
            ratio = len(r['generated_answer']) / len(r['reference_answer'])
            if ratio < 0.3 or ratio > 3.0:
                abnormal_ratios.append(r)

    if abnormal_ratios:
        validation['warnings'].append(f"Found {len(abnormal_ratios)} answers with abnormal length ratios")

    # Check metrics statistics
    metrics = ['bleu', 'f1_score', 'exact_match', 'rouge1']
    for metric in metrics:
        values = [r.get(metric, 0) for r in results]
        if values:
            validation['statistics'][metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'below_threshold': sum(1 for v in values if v < 0.3),
            }

    # Overall quality check
    low_quality = sum(1 for r in results if r.get('bleu', 0) < 0.3 and r.get('f1_score', 0) < 0.3)
    if low_quality > len(results) * 0.3:  # More than 30% are low quality
        validation['warnings'].append(f"High proportion of low quality answers: {low_quality} out of {len(results)}")

    return validation


if __name__ == "__main__":
    # If run directly, validate the environment
    results = validate_evaluation_environment()

    # Print validation results
    print("\nEnvironment Validation Results:")
    print(f"- Python version: {results['system']['python_version']}")
    print(f"- OS: {results['system']['os']}")

    print("\nDependencies:")
    for package, info in results['dependencies'].items():
        status = "✓" if info['installed'] else "✗"
        version = info.get('version', 'N/A')
        print(f"- {status} {package} ({version}): {info['purpose']}")

    print("\nGPU Availability:")
    if isinstance(results['gpu'], dict):
        if results['gpu']['available']:
            print(f"- GPU detected: {results['gpu']['device_name']}")
        else:
            print("- No GPU detected. Evaluation will be slower on CPU.")
    else:
        print(f"- {results['gpu']}")

    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"- {warning}")

    if results['critical_issues']:
        print("\nCritical Issues (must be resolved):")
        for issue in results['critical_issues']:
            print(f"- {issue}")
