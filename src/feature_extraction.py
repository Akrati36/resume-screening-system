import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FeatureExtractor:
    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        """
        Initialize feature extractor
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF (default: unigrams and bigrams)
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,
            max_df=0.9,
            sublinear_tf=True
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        
        self.is_fitted = False
    
    def fit_transform_tfidf(self, documents):
        """
        Fit TF-IDF vectorizer and transform documents
        
        Args:
            documents: List of text documents
            
        Returns:
            TF-IDF matrix
        """
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.is_fitted = True
        
        print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
        print(f"Vocabulary Size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return tfidf_matrix
    
    def transform_tfidf(self, documents):
        """
        Transform documents using fitted TF-IDF vectorizer
        
        Args:
            documents: List of text documents
            
        Returns:
            TF-IDF matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform_tfidf first.")
        
        return self.tfidf_vectorizer.transform(documents)
    
    def get_feature_names(self):
        """Get feature names from TF-IDF vectorizer"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
        
        return self.tfidf_vectorizer.get_feature_names_out()
    
    def get_top_features(self, document_vector, top_n=10):
        """
        Get top N features for a document
        
        Args:
            document_vector: TF-IDF vector for a document
            top_n: Number of top features to return
            
        Returns:
            List of (feature, score) tuples
        """
        feature_names = self.get_feature_names()
        
        # Get feature scores
        scores = document_vector.toarray()[0]
        
        # Get top indices
        top_indices = np.argsort(scores)[-top_n:][::-1]
        
        # Get top features and scores
        top_features = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        return top_features
    
    def calculate_cosine_similarity(self, vector1, vector2):
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vector1: First TF-IDF vector
            vector2: Second TF-IDF vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        similarity = cosine_similarity(vector1, vector2)[0][0]
        return similarity
    
    def calculate_similarity_matrix(self, matrix1, matrix2=None):
        """
        Calculate cosine similarity matrix
        
        Args:
            matrix1: First TF-IDF matrix
            matrix2: Second TF-IDF matrix (optional, if None uses matrix1)
            
        Returns:
            Similarity matrix
        """
        if matrix2 is None:
            matrix2 = matrix1
        
        similarity_matrix = cosine_similarity(matrix1, matrix2)
        return similarity_matrix
    
    def match_resume_to_job(self, resume_vector, job_vector):
        """
        Match a resume to a job description
        
        Args:
            resume_vector: TF-IDF vector of resume
            job_vector: TF-IDF vector of job description
            
        Returns:
            Match score (0-100)
        """
        similarity = self.calculate_cosine_similarity(resume_vector, job_vector)
        match_score = similarity * 100
        
        return round(match_score, 2)
    
    def rank_resumes(self, resume_vectors, job_vector):
        """
        Rank resumes based on similarity to job description
        
        Args:
            resume_vectors: TF-IDF matrix of resumes
            job_vector: TF-IDF vector of job description
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        similarities = cosine_similarity(resume_vectors, job_vector).flatten()
        scores = similarities * 100
        
        # Create ranking
        ranking = [(i, round(score, 2)) for i, score in enumerate(scores)]
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def get_vocabulary_stats(self):
        """Get statistics about the vocabulary"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
        
        vocab = self.tfidf_vectorizer.vocabulary_
        feature_names = self.get_feature_names()
        
        stats = {
            'vocabulary_size': len(vocab),
            'total_features': len(feature_names),
            'ngram_range': self.tfidf_vectorizer.ngram_range,
            'max_features': self.tfidf_vectorizer.max_features
        }
        
        return stats