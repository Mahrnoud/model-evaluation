import logging
import re
import unicodedata
from difflib import SequenceMatcher

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from metrics import calculate_rouge, calculate_f1_word_match, calculate_exact_match
from utils import clean_text_for_comparison

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


def normalize_arabic(text):
    """
    Enhanced Arabic text normalization that handles edge cases better

    Args:
        text: Input Arabic text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to string if not already
    text = str(text)

    # Normalize to Unicode form NFKC (more aggressive normalization)
    text = unicodedata.normalize('NFKC', text)

    # Normalize Alef variations
    text = re.sub("[إأآا]", "ا", text)

    # Normalize Hamzas
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)

    # Normalize Yeh and Alef Maksura
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)

    # Remove diacritics (all harakat)
    text = re.sub(r'[\u064B-\u0652\u0670]', '', text)

    # Remove tatweel (kashida)
    text = re.sub(r'\u0640', '', text)

    # Remove punctuation but keep spaces
    text = re.sub(r'[^\u0600-\u06FF\s\w]', '', text)

    # Normalize whitespace (all whitespace becomes a single space)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. Improve Arabic normalization with more complete character mapping
def enhanced_normalize_arabic(text):
    """
    More comprehensive Arabic text normalization with additional character mappings

    Args:
        text: Input Arabic text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to string if not already
    text = str(text)

    # Normalize to Unicode form NFKC (more aggressive normalization)
    text = unicodedata.normalize('NFKC', text)

    # More comprehensive Alef variations normalization
    text = re.sub("[إأآا\u0672\u0673\u0675]", "ا", text)

    # Normalize all types of Hamzas
    text = re.sub("[ؤئء]", "ء", text)

    # Normalize Yeh variations and Alef Maksura
    text = re.sub("[ىیيې]", "ي", text)
    text = re.sub("ة", "ه", text)

    # Normalize Waw variations
    text = re.sub("[ؤۆۇۉ]", "و", text)

    # Remove diacritics (all harakat) including additional marks
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)

    # Remove tatweel (kashida)
    text = re.sub(r'\u0640', '', text)

    # Normalize Persian/Urdu characters that might appear in Arabic text
    text = re.sub("پ", "ب", text)
    text = re.sub("چ", "ج", text)
    text = re.sub("ژ", "ز", text)
    text = re.sub("گ", "ك", text)

    # Normalize different forms of Kaf
    text = re.sub("[كک]", "ك", text)

    # Normalize Arabic numbers to standard digits if needed
    # Uncomment if you want this behavior
    # text = re.sub(r'[٠١٢٣٤٥٦٧٨٩]', lambda m: str("٠١٢٣٤٥٦٧٨٩".index(m.group())), text)

    # Remove punctuation but keep spaces
    text = re.sub(r'[^\u0600-\u06FF\s\w]', '', text)

    # Normalize whitespace (all whitespace becomes a single space)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def improved_tokenize_arabic(text):
    """
    Improved tokenization for Arabic text that works better for metrics

    Args:
        text: Arabic text to tokenize

    Returns:
        List of tokens
    """
    # First try to use pyarabic for better tokenization
    try:
        import pyarabic.araby as araby
        tokens = araby.tokenize(text)
        return tokens
    except ImportError:
        # Fallback to basic tokenization
        pass

    # Simple tokenization - split on whitespace and punctuation
    tokens = []
    current_token = ""

    for char in text:
        if '\u0600' <= char <= '\u06FF' or char.isalnum():
            # Arabic character or alphanumeric
            current_token += char
        else:
            # Non-Arabic, non-alphanumeric character
            if current_token:
                tokens.append(current_token)
                current_token = ""

            # Add non-space punctuation as separate tokens
            if not char.isspace():
                tokens.append(char)

    # Add the last token if there is one
    if current_token:
        tokens.append(current_token)

    return tokens


# 2. Add advanced Arabic tokenization with farasa if available
def advanced_tokenize_arabic(text):
    """
    Advanced tokenization for Arabic text using best available tools

    Args:
        text: Arabic text to tokenize

    Returns:
        List of tokens
    """
    # Try to use Farasa for state-of-the-art Arabic tokenization
    try:
        from farasapy.stemmer import FarasaStemmer
        stemmer = FarasaStemmer()
        tokens = stemmer.segment(text).split()
        return tokens
    except ImportError:
        # Fall back to CAMeL Tools if available
        try:
            from camel_tools.tokenizers.word import WordTokenizer
            tokenizer = WordTokenizer()
            tokens = tokenizer.tokenize(text)
            return tokens
        except ImportError:
            # Fall back to pyarabic
            try:
                import pyarabic.araby as araby
                tokens = araby.tokenize(text)
                return tokens
            except ImportError:
                # Fallback to basic tokenization as last resort
                pass

    # Simple tokenization as last resort
    tokens = []
    current_token = ""

    for char in text:
        if '\u0600' <= char <= '\u06FF' or char.isalnum():
            # Arabic character or alphanumeric
            current_token += char
        else:
            # Non-Arabic, non-alphanumeric character
            if current_token:
                tokens.append(current_token)
                current_token = ""

            # Add non-space punctuation as separate tokens
            if not char.isspace():
                tokens.append(char)

    # Add the last token if there is one
    if current_token:
        tokens.append(current_token)

    return tokens


def get_arabic_stopwords():
    """Get a set of Arabic stopwords, using camel_tools if available, or a basic set otherwise"""
    try:
        from camel_tools.utils.stopwords import StopWords
        return set(StopWords().STOPWORDS)
    except ImportError:
        # Basic set of common Arabic stopwords
        return set([
            'من', 'إلى', 'عن', 'على', 'في', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'و', 'ف', 'ثم', 'أو', 'أم', 'لكن', 'بل', 'لا', 'ما', 'ليس',
            'كان', 'يكون', 'أصبح', 'أضحى', 'ظل', 'بات', 'صار', 'لم', 'لن', 'لو',
            'إن', 'أن', 'كي', 'التي', 'الذي', 'اللذان', 'اللتان', 'الذين', 'اللاتي',
            'هو', 'هي', 'هم', 'هن', 'نحن', 'أنت', 'أنتما', 'أنتم', 'أنتن', 'أنا'
        ])


# 3. Add Arabic-specific BLEU scoring
def calculate_arabic_bleu(reference, hypothesis):
    """
    Calculate BLEU score with Arabic-specific considerations

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

    # Normalize Arabic texts
    reference = enhanced_normalize_arabic(reference)
    hypothesis = enhanced_normalize_arabic(hypothesis)

    # If they're equal after normalization, return 1.0
    if reference == hypothesis:
        return 1.0

    # Try word-level tokenization
    reference_tokens = advanced_tokenize_arabic(reference)
    hypothesis_tokens = advanced_tokenize_arabic(hypothesis)

    # If tokenization failed or produced empty results, fall back to character-level
    if not reference_tokens or not hypothesis_tokens:
        # Character-level tokenization as fallback
        reference_tokens = list(reference)
        hypothesis_tokens = list(hypothesis)

    # Use smoothing function to handle edge cases
    smoothie = SmoothingFunction().method1

    # Set weights for different n-grams (focus more on unigrams for Arabic)
    weights = (0.7, 0.3, 0.0, 0.0)  # More weight on unigrams for Arabic

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


# 4. Add support for dialectal Arabic
def detect_arabic_dialect(text):
    """
    Detect the dialect of Arabic text

    Args:
        text: Arabic text

    Returns:
        Detected dialect or "MSA" for Modern Standard Arabic
    """
    try:
        # Try to use dialect detection if available
        from dialectal_arabic import dialect_detector
        dialect = dialect_detector.detect(text)
        return dialect
    except ImportError:
        # Simple heuristic-based detection as fallback
        # These are basic indicators, not comprehensive

        egyptian_markers = ["احنا", "انتو", "بتاع", "دلوقتي", "عايز", "كدة", "مش"]
        levantine_markers = ["هيك", "هيدا", "شو", "بدي", "منيح", "فيني"]
        gulf_markers = ["وش", "يبغى", "طاري", "وايد", "عيل"]
        maghrebi_markers = ["واش", "زعمة", "بزاف", "كيفاش", "غادي"]

        text_lower = text.lower()

        for marker in egyptian_markers:
            if marker in text_lower:
                return "Egyptian"

        for marker in levantine_markers:
            if marker in text_lower:
                return "Levantine"

        for marker in gulf_markers:
            if marker in text_lower:
                return "Gulf"

        for marker in maghrebi_markers:
            if marker in text_lower:
                return "Maghrebi"

        # Default to MSA if no dialect markers are found
        return "MSA"


# 5. Add comprehensive Arabic text metrics
def calculate_arabic_metrics(reference, hypothesis):
    """
    Calculate comprehensive metrics for Arabic text evaluation

    Args:
        reference: Reference text
        hypothesis: Generated text

    Returns:
        Dictionary with multiple metrics
    """
    # Basic metrics
    bleu = calculate_arabic_bleu(reference, hypothesis)
    rouge_scores = calculate_rouge(reference, hypothesis)
    f1 = calculate_f1_word_match(reference, hypothesis, exclude_stopwords=True)
    exact_match = calculate_exact_match(reference, hypothesis)

    # Detect dialect
    ref_dialect = detect_arabic_dialect(reference)
    hyp_dialect = detect_arabic_dialect(hypothesis)
    dialect_match = 1.0 if ref_dialect == hyp_dialect else 0.0

    # Try to use more sophisticated Arabic NLP metrics if available
    try:
        from camel_tools.sentiment import SentimentAnalyzer

        # Sentiment analysis
        analyzer = SentimentAnalyzer.pretrained()
        ref_sentiment = analyzer.predict(reference)
        hyp_sentiment = analyzer.predict(hypothesis)
        sentiment_match = 1.0 if ref_sentiment == hyp_sentiment else 0.0
    except ImportError:
        sentiment_match = None

    # Combined metrics
    return {
        'bleu': bleu,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'f1_score': f1,
        'exact_match': exact_match,
        'dialect': {
            'reference': ref_dialect,
            'hypothesis': hyp_dialect,
            'match': dialect_match
        },
        'sentiment_match': sentiment_match
    }


def debug_arabic_processing(text):
    """
    Debug utility for Arabic text processing

    Args:
        text: Arabic text to debug

    Returns:
        Dictionary with detailed processing information
    """
    results = {
        'original': text,
        'length': len(text),
        'unicode_info': []
    }

    # Analyze characters
    for i, char in enumerate(text):
        results['unicode_info'].append({
            'position': i,
            'char': char,
            'unicode': f'U+{ord(char):04X}',
            'name': unicodedata.name(char, 'Unknown')
        })

    # Apply various processing steps
    results['processed'] = {
        'normalized_nfc': unicodedata.normalize('NFC', text),
        'normalized_nfkc': unicodedata.normalize('NFKC', text),
        'normalized_arabic': normalize_arabic(text),
        'cleaned': clean_text_for_comparison(text),
    }

    # Tokenization results
    results['tokenization'] = {
        'basic_split': text.split(),
        'improved_tokenize': improved_tokenize_arabic(text),
        'advanced_tokenize': advanced_tokenize_arabic(text) if 'advanced_tokenize_arabic' in globals() else None
    }

    return results
