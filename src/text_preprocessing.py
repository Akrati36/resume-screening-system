import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, use_stemming=False, use_lemmatization=True):
        """
        Initialize text preprocessor
        
        Args:
            use_stemming: Whether to use stemming
            use_lemmatization: Whether to use lemmatization
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
    def clean_text(self, text):
        """Remove special characters, URLs, emails, etc."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\+?\d[\d -]{8,}\d', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens):
        """Stem tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text, return_string=True):
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            return_string: If True, return processed text as string, else return tokens
            
        Returns:
            Preprocessed text (string or list of tokens)
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply stemming or lemmatization
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        elif self.use_stemming:
            tokens = self.stem_tokens(tokens)
        
        # Return as string or tokens
        if return_string:
            return ' '.join(tokens)
        return tokens
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in texts]
    
    def extract_skills(self, text, skill_keywords):
        """
        Extract skills from text based on keyword list
        
        Args:
            text: Input text
            skill_keywords: List of skill keywords to search for
            
        Returns:
            List of found skills
        """
        text_lower = text.lower()
        found_skills = []
        
        for skill in skill_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_experience_years(self, text):
        """Extract years of experience from text"""
        # Pattern for "X years of experience" or "X+ years"
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp)\s*(?:of\s*)?(\d+)\+?\s*(?:years?|yrs?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        return 0
    
    def extract_education(self, text):
        """Extract education qualifications from text"""
        education_keywords = [
            'phd', 'ph.d', 'doctorate',
            'master', 'mba', 'ms', 'm.s', 'ma', 'm.a',
            'bachelor', 'bs', 'b.s', 'ba', 'b.a', 'btech', 'b.tech',
            'diploma', 'associate'
        ]
        
        text_lower = text.lower()
        found_education = []
        
        for edu in education_keywords:
            if edu in text_lower:
                found_education.append(edu)
        
        return list(set(found_education))  # Remove duplicates