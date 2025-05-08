"""
Test suite for QA evaluation framework.

This module contains unit tests, integration tests, and property-based tests
for the QA evaluation framework, with special focus on Arabic language support.
"""

import os
import tempfile
import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from ..language.arabic import normalize_arabic, enhanced_normalize_arabic, improved_tokenize_arabic
from ..metrics.bleu import calculate_bleu
from ..metrics.rouge import calculate_rouge
from ..metrics.exact_match import calculate_exact_match, calculate_f1_word_match
from ..evaluation.evaluator import evaluate_qa_model

logger = logging.getLogger(__name__)


def test_arabic_normalization():
    """Test Arabic normalization function"""
    # Test cases with various Arabic text variations
    test_cases = [
        # Original, Expected normalized
        ("أهلاً بالعالم", "اهلا بالعالم"),  # Basic case with diacritics
        ("إنَّ السَّماءَ صافيةٌ", "ان السماء صافيه"),  # Multiple normalizations
        ("هَذِهِ المَقالَةُ رائِعَةٌ", "هذه المقاله رايعه"),  # More diacritics
        ("أُحِبُّ اللُّغَةَ العَرَبِيَّةَ", "احب اللغه العربيه"),  # Complex case
        ("محمّد", "محمد"),  # Shadda removal
        ("العَرَبِيَّة", "العربيه"),  # Combined cases
    ]

    for original, expected in test_cases:
        normalized = normalize_arabic(original)
        assert normalized == expected, f"Failed for: {original}\nExpected: {expected}\nGot: {normalized}"

    # Test the enhanced version if available
    if 'enhanced_normalize_arabic' in globals():
        for original, expected in test_cases:
            normalized = enhanced_normalize_arabic(original)
            assert normalized == expected, f"Enhanced normalization failed for: {original}\nExpected: {expected}\nGot: {normalized}"


def test_calculate_bleu():
    """Test BLEU score calculation"""
    # Test pairs of references and hypotheses
    test_cases = [
        # Reference, Hypothesis, Expected approximate score
        ("This is a test", "This is a test", 1.0),  # Exact match
        ("This is a test", "This is the test", 0.7),  # One word different
        ("This is a test", "That was a test", 0.5),  # Two words different
        ("This is a test", "Something completely different", 0.0),  # All different
        # Arabic examples
        ("هذا اختبار", "هذا اختبار", 1.0),  # Exact match
        ("هذا اختبار بسيط", "هذا اختبار", 0.7),  # Partial match
    ]

    for reference, hypothesis, expected in test_cases:
        score = calculate_bleu(reference, hypothesis)
        # Allow for some wiggle room in the score due to tokenization differences
        assert abs(
            score - expected) < 0.3, f"Failed for: {reference} / {hypothesis}\nExpected ~{expected}\nGot: {score}"


def test_exact_match():
    """Test exact match calculation with various edge cases"""
    test_cases = [
        # Reference, Hypothesis, Expected score
        ("Test", "Test", 1.0),  # Simple exact match
        ("Test", "test", 0.95),  # Case difference
        ("Test", "Test!", 0.0),  # Punctuation difference
        ("Test  string", "Test string", 1.0),  # Whitespace normalization
        # Arabic cases
        ("اختبار", "اختبار", 1.0),  # Simple exact match
        ("اختبار", "إختبار", 1.0),  # Alef normalization
        ("إنَّ السَّماءَ", "ان السماء", 1.0),  # With diacritics
    ]

    for reference, hypothesis, expected in test_cases:
        score = calculate_exact_match(reference, hypothesis)
        assert abs(
            score - expected) < 0.1, f"Failed for: {reference} / {hypothesis}\nExpected: {expected}\nGot: {score}"


def test_rouge():
    """Test ROUGE score calculation"""
    # Test cases for ROUGE scoring
    test_cases = [
        # Reference, Hypothesis, Expected ROUGE-L
        ("This is a test sentence", "This is a test sentence", 1.0),  # Exact match
        ("This is a test sentence", "This is a test", 0.8),  # Shorter
        ("This is a test", "This is a test sentence", 0.8),  # Longer
        ("This is a test sentence", "This is a different sentence", 0.6),  # Similar
        # Arabic examples
        ("هذا اختبار للنص العربي", "هذا اختبار للنص العربي", 1.0),  # Exact match
        ("هذا اختبار للنص العربي", "هذه جملة اختبار", 0.3),  # Different
    ]

    for reference, hypothesis, expected in test_cases:
        scores = calculate_rouge(reference, hypothesis)
        rouge_l = scores['rougeL']
        assert abs(
            rouge_l - expected) < 0.3, f"Failed for: {reference} / {hypothesis}\nExpected ~{expected}\nGot: {rouge_l}"


def test_f1_word_match():
    """Test F1 word match calculation"""
    # Test cases for F1 word match
    test_cases = [
        # Reference, Hypothesis, Expected F1
        ("The quick brown fox", "The quick brown fox", 1.0),  # Exact match
        ("The quick brown fox", "The quick fox", 0.75),  # Missing word
        ("The fox", "The quick brown fox", 0.5),  # Additional words
        # Arabic examples
        ("الثعلب البني السريع", "الثعلب البني السريع", 1.0),  # Exact match
        ("الثعلب البني السريع", "الثعلب السريع", 0.8),  # Missing word
    ]

    for reference, hypothesis, expected in test_cases:
        score = calculate_f1_word_match(reference, hypothesis)
        assert abs(
            score - expected) < 0.3, f"Failed for: {reference} / {hypothesis}\nExpected ~{expected}\nGot: {score}"


def test_tokenization():
    """Test Arabic tokenization function"""
    test_cases = [
        # Input text, Expected token count
        ("هذا اختبار بسيط", 3),  # Simple case
        ("هل تعمل وظيفة التجزئة؟", 4),  # Question with punctuation
        ("نص عربي مع علامات؟ ترقيم!", 6),  # Multiple sentences
    ]

    for text, expected_count in test_cases:
        tokens = improved_tokenize_arabic(text)
        assert len(
            tokens) == expected_count, f"Failed for: {text}\nExpected {expected_count} tokens, got {len(tokens)}: {tokens}"


@patch('greedy_inference.load_model_and_tokenizer')
@patch('greedy_inference.greedy_decode')
def test_evaluation_flow(mock_greedy_decode, mock_load_model):
    """Test the full evaluation flow with mocked model"""
    from ..evaluation.config import EvaluationConfig

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock data
        mock_data = [
            {"question": "[LANG_EN] What is testing?", "reference_answer": "Testing is verification.",
             "category": "General"},
            {"question": "[LANG_EN] How to test?", "reference_answer": "Use unit tests.", "category": "Programming"},
        ]

        # Create a temporary dataset file
        dataset_path = os.path.join(temp_dir, "test_dataset.json")
        with open(dataset_path, 'w') as f:
            json.dump(mock_data, f)

        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer, "cpu")

        # Mock the greedy_decode function to return simple answers
        mock_greedy_decode.side_effect = ["Testing is verification.", "Write test cases."]

        # Create evaluation config
        config = EvaluationConfig(
            model_path="dummy_path",
            tokenizer_path="dummy_tokenizer",
            dataset_path=dataset_path,
            question_lang="en",
            output_dir=temp_dir,
            sample_size=None,
            max_length=100,
            debug_mode=True,
            visualize=False,
            html_report=False
        )

        # Run evaluation
        summary = evaluate_qa_model(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            config=config
        )

        # Assertions on the evaluation results
        assert summary is not None
        assert summary['total_samples'] == 2
        assert summary['language'] == "en"
        assert 'average_metrics' in summary
        assert 'category_metrics' in summary
        assert 'General' in summary['category_metrics']
        assert 'Programming' in summary['category_metrics']

        # Check if output files were created
        assert os.path.exists(os.path.join(temp_dir, "detailed_results_en.json"))
        assert os.path.exists(os.path.join(temp_dir, "summary_en.json"))


# Try to add property-based tests with hypothesis if available
try:
    from hypothesis import given, strategies as st

    # Define strategies for text
    ascii_text = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=127), min_size=1)
    arabic_text = st.text(alphabet=st.characters(min_codepoint=0x0600, max_codepoint=0x06FF), min_size=1)


    @given(ascii_text, ascii_text)
    def test_bleu_properties(text1, text2):
        """Test mathematical properties of BLEU score"""
        # BLEU should be between 0 and 1
        score = calculate_bleu(text1, text2)
        assert 0 <= score <= 1

        # Symmetric property: if texts are swapped, score should be similar
        reverse_score = calculate_bleu(text2, text1)
        assert abs(score - reverse_score) < 0.5  # Allow for some asymmetry

        # Identity property: text compared with itself should have score 1
        if text1:
            identity_score = calculate_bleu(text1, text1)
            assert identity_score > 0.9  # Should be very close to 1


    @given(arabic_text)
    def test_arabic_normalization_idempotent(text):
        """Test that Arabic normalization is idempotent (running it twice gives same result)"""
        once = normalize_arabic(text)
        twice = normalize_arabic(once)
        assert once == twice
except ImportError:
    # If hypothesis is not installed, skip these tests
    logger.warning("Hypothesis not installed, skipping property-based tests")
    pass


def run_tests():
    """Run the test suite for the QA evaluation code"""
    # Try to use unittest framework if available
    try:
        # Define a test class for unittest
        class QAEvaluationTests(unittest.TestCase):
            def test_arabic_normalization(self):
                test_arabic_normalization()

            def test_calculate_bleu(self):
                test_calculate_bleu()

            def test_exact_match(self):
                test_exact_match()

            def test_rouge(self):
                test_rouge()

            def test_f1_word_match(self):
                test_f1_word_match()

            def test_tokenization(self):
                test_tokenization()

            def test_evaluation_flow(self):
                with patch('greedy_inference.load_model_and_tokenizer'), \
                        patch('greedy_inference.greedy_decode'):
                    test_evaluation_flow(None, None)

        # Run the tests using unittest
        unittest.main(argv=['first-arg-is-ignored'], exit=False)

    except Exception as e:
        # Fall back to simple testing
        print(f"Error setting up unittest: {e}")
        print("Running basic tests...")
        try:
            test_arabic_normalization()
            print("✓ Arabic normalization tests passed")
        except Exception as e:
            print(f"✗ Arabic normalization tests failed: {e}")

        try:
            test_calculate_bleu()
            print("✓ BLEU calculation tests passed")
        except Exception as e:
            print(f"✗ BLEU calculation tests failed: {e}")

        try:
            test_exact_match()
            print("✓ Exact match tests passed")
        except Exception as e:
            print(f"✗ Exact match tests failed: {e}")

        try:
            test_rouge()
            print("✓ ROUGE calculation tests passed")
        except Exception as e:
            print(f"✗ ROUGE calculation tests failed: {e}")

        try:
            test_f1_word_match()
            print("✓ F1 word match tests passed")
        except Exception as e:
            print(f"✗ F1 word match tests failed: {e}")

        try:
            test_tokenization()
            print("✓ Tokenization tests passed")
        except Exception as e:
            print(f"✗ Tokenization tests failed: {e}")


if __name__ == "__main__":
    run_tests()