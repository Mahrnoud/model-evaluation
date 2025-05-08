import json
import logging

import pandas as pd

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


def load_dataset(dataset_path, sample_size, question_lang):
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except:
        # Try loading from JSON if CSV fails
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")

    # Sample data if needed
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)

    # Map language codes to column names or extract from data structure
    if question_lang == 'en':
        # Try different ways to get questions and answers
        if 'question' in df.columns and 'en' in df['question'].iloc[0]:
            # Question column contains language tag
            questions = df['question'].apply(lambda x: x if '[LANG_EN]' in x else f'[LANG_EN] {x}')
            try:
                answers = df['reference_answer']
            except:
                answers = df['answer'] if 'answer' in df.columns else df['generated_answer']
        else:
            # Try to find appropriate columns
            question_col = next((col for col in df.columns if 'en' in col.lower() and 'question' in col.lower()), None)
            answer_col = next((col for col in df.columns if
                               'en' in col.lower() and any(x in col.lower() for x in ['answer', 'reference'])), None)

            if not question_col or not answer_col:
                # Assume standard columns
                question_col = 'question'
                answer_col = 'reference_answer' if 'reference_answer' in df.columns else 'answer'

            questions = df[question_col].apply(lambda x: x if '[LANG_EN]' in x else f'[LANG_EN] {x}')
            answers = df[answer_col]
    else:  # Arabic
        # Try different ways to get questions and answers
        if 'question' in df.columns and 'ar' in df['question'].iloc[0]:
            # Question column contains language tag
            questions = df['question'].apply(lambda x: x if '[LANG_AR]' in x else f'[LANG_AR] {x}')
            try:
                answers = df['reference_answer']
            except:
                answers = df['answer'] if 'answer' in df.columns else df['generated_answer']
        else:
            # Try to find appropriate columns
            question_col = next((col for col in df.columns if 'ar' in col.lower() and 'question' in col.lower()), None)
            answer_col = next((col for col in df.columns if
                               'ar' in col.lower() and any(x in col.lower() for x in ['answer', 'reference'])), None)

            if not question_col or not answer_col:
                # Assume standard columns
                question_col = 'question'
                answer_col = 'reference_answer' if 'reference_answer' in df.columns else 'answer'

            questions = df[question_col].apply(lambda x: x if '[LANG_AR]' in x else f'[LANG_AR] {x}')
            answers = df[answer_col]

    # Get categories if available
    if 'category' in df.columns:
        categories = df['category']
        sub_categories = df['sub_category'] if 'sub_category' in df.columns else [None] * len(df)
    else:
        categories = ['Unknown'] * len(df)
        sub_categories = [None] * len(df)

    return questions, answers, categories, sub_categories


def save_results():
    pass
