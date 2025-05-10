# Model Evaluation Framework

A comprehensive evaluation framework for question-answering models with enhanced Arabic language support.

## Overview

This framework provides tools for evaluating the performance of question-answering models, with special focus on Arabic language processing and multilingual evaluation metrics. It includes support for various evaluation metrics, visualization tools, and detailed error analysis.

## Features

- **Multilingual Support**: Evaluate QA models on both English and Arabic datasets
- **Enhanced Arabic Processing**: Specialized normalization, tokenization, and metrics for Arabic text
- **Comprehensive Metrics**: BLEU, ROUGE, F1, Exact Match, and custom metrics
- **Visualization Tools**: Generate visual reports and comparisons across languages
- **Human Evaluation Support**: Create spreadsheets for manual review of model outputs
- **Detailed Error Analysis**: Identify common error patterns and edge cases
- **Environment Validation**: Verify dependencies and GPU availability

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- NLTK
- pandas
- matplotlib
- seaborn (optional, for enhanced visualizations)
- rouge_score
- tqdm

### Optional Dependencies for Arabic Support

- pyarabic
- camel_tools
- farasapy

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Mahrnoud/model-evaluation.git
   cd model-evaluation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install optional Arabic support:
   ```bash
   pip install pyarabic camel_tools farasapy
   ```

4. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

### Basic Evaluation

To evaluate a model on a dataset:

```bash
python main.py --model_path path/to/model \
               --tokenizer_path path/to/tokenizer \
               --dataset_path path/to/dataset.csv \
               --question_lang en
```

### Multilingual Evaluation

To evaluate on both English and Arabic:

```bash
python main.py --model_path path/to/model \
               --tokenizer_path path/to/tokenizer \
               --dataset_path path/to/dataset.csv \
               --mode both
```

### Comparative Analysis

For detailed language comparison:

```bash
python main.py --model_path path/to/model \
               --tokenizer_path path/to/tokenizer \
               --dataset_path path/to/dataset.csv \
               --mode compare \
               --html_report
```

### Environment Validation

To check if all dependencies are correctly installed:

```bash
python main.py --validate
```

### Stress Testing

To determine optimal batch size and maximum length for your GPU:

```bash
python main.py --model_path path/to/model \
               --tokenizer_path path/to/tokenizer \
               --stress_test
```

## Project Structure

- **`language/`**: Language-specific processing modules
  - `arabic.py`: Arabic text normalization and tokenization
  - `english.py`: English text cleaning utilities

- **`metrics/`**: Evaluation metrics implementations
  - `bleu.py`: BLEU score calculation
  - `rouge.py`: ROUGE metrics
  - `exact_match.py`: Exact match and F1 word matching

- **`utils/`**: Utility functions
  - `text_processing.py`: Text cleaning and preprocessing
  - `io.py`: Dataset loading and result saving
  - `greedy_inference.py`: Model inference utilities

- **`evaluation/`**: Evaluation framework
  - `evaluator.py`: Main evaluation functions
  - `config.py`: Configuration management

- **`visualization/`**: Reporting and visualization tools
  - `plots.py`: Generate visualizations
  - `reports.py`: Create HTML reports

- **`testing/`**: Test and validation utilities
  - `test_suite.py`: Unit tests for framework components
  - `validation.py`: Environment validation

## Dataset Format

The framework supports CSV and JSON datasets with the following expected columns:
- `question`: The input question (with language tags like `[LANG_EN]` or `[LANG_AR]`)
- `reference_answer`: The expected answer
- `category` (optional): Question category for fine-grained analysis
- `sub_category` (optional): Question sub-category

## Output Files

The framework generates several output files in the specified output directory:
- `detailed_results_[lang].json`: Complete evaluation results
- `summary_[lang].json`: Summary statistics
- `[lang]_quality_distribution.png`: Visualization of quality distribution
- `[lang]_score_distributions.png`: Distributions of evaluation metrics
- `[lang]_evaluation_report.html`: Comprehensive HTML report
- `bilingual_comparison_report.html`: Comparative analysis across languages
- `human_evaluation_[lang].xlsx`: Spreadsheet for manual review

## Arabic Language Support

This framework includes specialized processing for Arabic text:

1. **Normalization**: Handles Arabic-specific character variations and diacritics
2. **Tokenization**: Multiple tokenization methods with fallbacks
3. **Dialect Detection**: Basic detection of Arabic dialects
4. **Metrics**: Adapted metrics that account for Arabic language characteristics

## Extending the Framework

### Adding New Metrics

To add a new evaluation metric, create a new module in the `metrics/` directory and update the imports in `metrics/__init__.py`.

### Supporting Additional Languages

To add support for a new language, create a language-specific module in the `language/` directory following the pattern of `arabic.py`.

### Custom Visualizations

To create custom visualizations, extend the functions in `visualization/plots.py` and call them from the evaluator.

## License

[MIT License](LICENSE)

## Citation

If you use this framework in your research, please cite:

```
@software{qa_evaluation_framework,
  author = {Mahmoud},
  title = {QA Evaluation Framework with Enhanced Arabic Support},
  year = {2025},
  url = {https://github.com/Mahrnoud/model-evaluation}
}
```

## Acknowledgments

- This framework uses the greedy inference model from [project reference]
- Arabic NLP support is built upon tools from [references to Arabic NLP tools]