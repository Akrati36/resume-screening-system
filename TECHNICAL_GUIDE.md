# 📚 Technical Guide - Resume Screening System

## Table of Contents
1. [NLP Preprocessing](#nlp-preprocessing)
2. [TF-IDF Vectorization](#tf-idf-vectorization)
3. [Cosine Similarity](#cosine-similarity)
4. [Classification Models](#classification-models)
5. [Performance Optimization](#performance-optimization)

---

## NLP Preprocessing

### What is NLP Preprocessing?

NLP preprocessing transforms raw text into a clean, standardized format suitable for machine learning.

### Pipeline Steps

#### 1. Text Cleaning
```python
# Original
"Python Developer with 5+ years experience! Email: john@email.com"

# After cleaning
"python developer years experience"
```

**Operations:**
- Convert to lowercase
- Remove URLs, emails, phone numbers
- Remove special characters and digits
- Remove extra whitespace

#### 2. Tokenization
```python
# Input
"python developer years experience"

# Output
["python", "developer", "years", "experience"]
```

**Purpose:** Break text into individual words (tokens)

#### 3. Stopword Removal
```python
# Input
["python", "developer", "with", "years", "of", "experience"]

# Output (stopwords removed)
["python", "developer", "years", "experience"]
```

**Stopwords:** Common words with little meaning (the, is, and, with, of, etc.)

#### 4. Lemmatization
```python
# Input
["running", "better", "studies", "developed"]

# Output (lemmatized)
["run", "good", "study", "develop"]
```

**Purpose:** Reduce words to their base/dictionary form

### Code Example

```python
from src.text_preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(use_lemmatization=True)

text = "I am a Python Developer with 5+ years of experience in ML!"
processed = preprocessor.preprocess(text)

print(processed)
# Output: "python developer year experience ml"
```

---

## TF-IDF Vectorization

### What is TF-IDF?

**TF-IDF** = Term Frequency × Inverse Document Frequency

It converts text into numerical vectors, giving higher weights to important, unique terms.

### Components

#### 1. Term Frequency (TF)
How often a term appears in a document.

```
TF(term, document) = (Number of times term appears) / (Total terms in document)
```

**Example:**
```
Document: "python python java"
TF(python) = 2/3 = 0.67
TF(java) = 1/3 = 0.33
```

#### 2. Inverse Document Frequency (IDF)
How unique a term is across all documents.

```
IDF(term) = log(Total documents / Documents containing term)
```

**Example:**
```
3 documents total
"python" appears in 2 documents
IDF(python) = log(3/2) = 0.18

"tensorflow" appears in 1 document
IDF(tensorflow) = log(3/1) = 0.48
```

#### 3. TF-IDF Score
```
TF-IDF = TF × IDF
```

**Interpretation:**
- High TF-IDF = Important, unique term
- Low TF-IDF = Common, less important term

### N-grams

**Unigrams:** Single words
```
"machine learning" → ["machine", "learning"]
```

**Bigrams:** Two-word phrases
```
"machine learning" → ["machine", "learning", "machine learning"]
```

**Why use bigrams?**
- Captures phrases like "machine learning", "data science"
- Better context understanding

### Code Example

```python
from src.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(max_features=500, ngram_range=(1, 2))

documents = [
    "python developer machine learning",
    "java backend developer",
    "data scientist python machine learning"
]

# Create TF-IDF matrix
tfidf_matrix = extractor.fit_transform_tfidf(documents)

print(tfidf_matrix.shape)  # (3, 500) - 3 documents, 500 features

# Get top features for first document
top_features = extractor.get_top_features(tfidf_matrix[0], top_n=5)
print(top_features)
# [('machine learning', 0.65), ('python', 0.45), ('developer', 0.40), ...]
```

---

## Cosine Similarity

### What is Cosine Similarity?

Measures the similarity between two vectors by calculating the cosine of the angle between them.

### Formula

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = magnitude of vector A
- ||B|| = magnitude of vector B
```

### Range
- **1.0** = Identical vectors (0° angle)
- **0.0** = Orthogonal vectors (90° angle)
- **-1.0** = Opposite vectors (180° angle)

### Visual Example

```
Resume Vector:    [0.5, 0.8, 0.3]  (python, ML, java)
Job Vector:       [0.6, 0.7, 0.1]  (python, ML, react)

Similarity = 0.92 (92% match)
```

### Why Cosine Similarity?

✅ **Advantages:**
- Ignores document length
- Focuses on content similarity
- Fast computation
- Works well with sparse vectors (TF-IDF)

❌ **Limitations:**
- Doesn't consider word order
- Treats all dimensions equally

### Code Example

```python
from src.feature_extraction import FeatureExtractor

extractor = FeatureExtractor()

# Job description
job = "Looking for Python developer with ML experience"
job_processed = preprocessor.preprocess(job)
job_vector = extractor.transform_tfidf([job_processed])

# Resume
resume = "Python developer with 5 years ML and data science experience"
resume_processed = preprocessor.preprocess(resume)
resume_vector = extractor.transform_tfidf([resume_processed])

# Calculate similarity
similarity = extractor.calculate_cosine_similarity(resume_vector, job_vector)
match_score = similarity * 100

print(f"Match Score: {match_score}%")  # 85.3%
```

---

## Classification Models

### 1. Logistic Regression

#### How it Works
Linear model with sigmoid activation for binary/multi-class classification.

```
P(class) = 1 / (1 + e^(-z))
where z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

#### Strengths
✅ Fast training and prediction
✅ Interpretable (feature weights)
✅ Works well with high-dimensional data (TF-IDF)
✅ Probabilistic output

#### Weaknesses
❌ Assumes linear decision boundary
❌ May underfit complex patterns

#### Code Example

```python
from src.resume_classifier import ResumeClassifier

classifier = ResumeClassifier()

# Train
lr_model = classifier.train_logistic_regression(X_train, y_train)

# Predict
prediction = classifier.classify_resume(lr_model, resume_vector)
print(prediction)
# {'predicted_class': 'Data Science', 'confidence': 87.5}

# Feature importance
importance = classifier.get_feature_importance(lr_model, feature_names)
```

### 2. Naive Bayes

#### How it Works
Probabilistic classifier based on Bayes' theorem with "naive" independence assumption.

```
P(class|features) = P(features|class) × P(class) / P(features)
```

**Naive Assumption:** Features are independent given the class.

#### Strengths
✅ Very fast training
✅ Works well with small datasets
✅ Handles high-dimensional data
✅ Good for text classification

#### Weaknesses
❌ Independence assumption rarely true
❌ Can be overconfident in predictions

#### Code Example

```python
# Train
nb_model = classifier.train_naive_bayes(X_train, y_train)

# Predict
prediction = classifier.classify_resume(nb_model, resume_vector)
```

### Model Comparison

| Metric | Logistic Regression | Naive Bayes |
|--------|-------------------|-------------|
| Training Speed | Fast | Very Fast |
| Prediction Speed | Fast | Very Fast |
| Accuracy | 85-90% | 80-85% |
| Interpretability | High | Medium |
| Overfitting Risk | Low | Low |
| Best For | Balanced datasets | Text classification |

---

## Performance Optimization

### 1. TF-IDF Optimization

```python
# Reduce vocabulary size
extractor = FeatureExtractor(
    max_features=500,      # Limit to top 500 features
    min_df=2,              # Ignore terms in < 2 documents
    max_df=0.8             # Ignore terms in > 80% documents
)
```

### 2. Preprocessing Optimization

```python
# Use stemming instead of lemmatization (faster)
preprocessor = TextPreprocessor(
    use_stemming=True,
    use_lemmatization=False
)
```

### 3. Batch Processing

```python
# Process multiple resumes at once
resumes = ["resume1", "resume2", "resume3"]
processed = preprocessor.preprocess_batch(resumes)
vectors = extractor.transform_tfidf(processed)
```

### 4. Model Selection

**For Speed:**
- Use Naive Bayes
- Reduce max_features
- Use stemming

**For Accuracy:**
- Use Logistic Regression
- Increase max_features
- Use lemmatization
- Add bigrams/trigrams

### 5. Caching

```python
import joblib

# Save vectorizer
joblib.dump(extractor.tfidf_vectorizer, 'vectorizer.pkl')

# Load vectorizer (avoid refitting)
vectorizer = joblib.load('vectorizer.pkl')
```

---

## Evaluation Metrics

### 1. Accuracy
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**When to use:** Balanced datasets

### 2. Precision
```
Precision = True Positives / (True Positives + False Positives)
```

**Interpretation:** Of all predicted positives, how many are correct?

### 3. Recall
```
Recall = True Positives / (True Positives + False Negatives)
```

**Interpretation:** Of all actual positives, how many did we find?

### 4. F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation:** Harmonic mean of precision and recall

### 5. ROC-AUC
Area under the Receiver Operating Characteristic curve.

**Range:** 0.5 (random) to 1.0 (perfect)

---

## Best Practices

### 1. Data Preparation
✅ Clean and normalize text
✅ Remove duplicates
✅ Balance classes (if needed)
✅ Split train/test properly

### 2. Feature Engineering
✅ Experiment with n-gram ranges
✅ Tune max_features
✅ Consider domain-specific stopwords
✅ Extract custom features (skills, experience)

### 3. Model Training
✅ Use cross-validation
✅ Try multiple models
✅ Tune hyperparameters
✅ Monitor overfitting

### 4. Evaluation
✅ Use multiple metrics
✅ Test on unseen data
✅ Analyze errors
✅ Get domain expert feedback

---

## Further Reading

- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn TF-IDF Guide](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

---

**Questions?** Open an issue on GitHub or check the [README](README.md)!