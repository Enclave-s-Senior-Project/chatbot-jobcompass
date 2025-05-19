# app/utils/nltk_setup.py
import nltk
import logging

logger = logging.getLogger(__name__)


def setup_nltk_data():
    """Download NLTK data only if not already present."""
    nltk_data = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, resource_name in nltk_data:
        try:
            nltk.data.find(resource_path)
            logger.info(f"NLTK resource {resource_name} already exists.")
        except LookupError:
            logger.info(f"Downloading NLTK resource {resource_name}...")
            nltk.download(resource_name, quiet=True)
