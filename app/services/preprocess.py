import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re

# Initialize stopwords with custom job-related terms
stop_words = set(stopwords.words('english'))
job_stopwords = {'job', 'position', 'company', 'work', 'team', 'skill', 'opportunity', 'role', 'industry', 'career'}
stop_words.update(job_stopwords)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    """Preprocess text by cleaning HTML, tokenizing, removing stopwords, and lemmatizing."""
    if not isinstance(text, str):
        return ""
    
    # Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Convert to lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Lemmatize with POS tagging, remove stopwords and short tokens
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) 
              for token in tokens if token not in stop_words and len(token) > 3]
    
    return ' '.join(tokens)