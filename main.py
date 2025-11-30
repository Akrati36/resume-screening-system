import pandas as pd
import numpy as np
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from resume_classifier import ResumeClassifier
from resume_parser import ResumeParser

def load_sample_data():
    """
    Load or create sample resume dataset
    For demonstration - replace with your actual dataset
    """
    # Sample data structure
    data = {
        'resume_text': [
            "Python developer with 5 years experience in machine learning and data science. Skilled in TensorFlow, PyTorch, scikit-learn. Built recommendation systems and NLP models.",
            "Java software engineer with expertise in Spring Boot, microservices, and cloud computing. 3 years experience in backend development and REST APIs.",
            "Data analyst proficient in SQL, Excel, Tableau. Experience in business intelligence and data visualization. Strong analytical skills.",
            "Full stack developer with React, Node.js, MongoDB experience. Built scalable web applications. 4 years in software development.",
            "Machine learning engineer specializing in computer vision and deep learning. Experience with CNNs, object detection, image segmentation.",
            "DevOps engineer with AWS, Docker, Kubernetes expertise. CI/CD pipeline automation. Infrastructure as code with Terraform.",
            "Frontend developer skilled in React, Vue.js, TypeScript. Responsive web design and modern JavaScript frameworks.",
            "Data scientist with PhD in statistics. Expert in predictive modeling, A/B testing, and experimental design. Python and R programming.",
            "Backend developer with expertise in Python Django, PostgreSQL, Redis. RESTful API design and database optimization.",
            "Mobile app developer with React Native and Flutter experience. Published apps on iOS and Android platforms."
        ],
        'category': [
            'Data Science', 'Software Engineering', 'Data Analysis', 'Software Engineering', 'Data Science',
            'DevOps', 'Software Engineering', 'Data Science', 'Software Engineering', 'Software Engineering'
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def main():
    """Complete Resume Screening System Pipeline"""
    
    print("="*70)
    print("RESUME SCREENING SYSTEM - NLP & ML")
    print("="*70)
    
    # Initialize components
    preprocessor = TextPreprocessor(use_lemmatization=True)
    feature_extractor = FeatureExtractor(max_features=500, ngram_range=(1, 2))
    classifier = ResumeClassifier()
    parser = ResumeParser()
    
    # Step 1: Load Data
    print("\n[STEP 1] Loading Resume Data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} resumes")
    print(f"\nCategory Distribution:")
    print(df['category'].value_counts())
    
    # Step 2: Text Preprocessing
    print("\n[STEP 2] Preprocessing Resume Text...")
    print("- Tokenization")
    print("- Stopword removal")
    print("- Lemmatization")
    
    df['processed_text'] = preprocessor.preprocess_batch(df['resume_text'])
    
    print("\nExample - Original vs Processed:")
    print(f"Original: {df['resume_text'].iloc[0][:100]}...")
    print(f"Processed: {df['processed_text'].iloc[0][:100]}...")
    
    # Step 3: TF-IDF Vectorization
    print("\n[STEP 3] TF-IDF Vectorization...")
    tfidf_matrix = feature_extractor.fit_transform_tfidf(df['processed_text'])
    
    # Show vocabulary stats
    vocab_stats = feature_extractor.get_vocabulary_stats()
    print(f"\nVocabulary Statistics:")
    for key, value in vocab_stats.items():
        print(f"  {key}: {value}")
    
    # Show top features for first resume
    print("\nTop 10 TF-IDF features for first resume:")
    top_features = feature_extractor.get_top_features(tfidf_matrix[0], top_n=10)
    for feature, score in top_features:
        print(f"  {feature}: {score:.4f}")
    
    # Step 4: Cosine Similarity Example
    print("\n[STEP 4] Cosine Similarity Matching...")
    
    # Example job description
    job_description = "Looking for a data scientist with Python, machine learning, and NLP experience. Must know TensorFlow and scikit-learn."
    job_processed = preprocessor.preprocess(job_description)
    job_vector = feature_extractor.transform_tfidf([job_processed])
    
    print(f"\nJob Description: {job_description}")
    print("\nMatching resumes to job description...")
    
    # Rank resumes
    rankings = feature_extractor.rank_resumes(tfidf_matrix, job_vector)
    
    print("\nTop 5 Matching Resumes:")
    for i, (idx, score) in enumerate(rankings[:5], 1):
        print(f"{i}. Resume {idx} - Match Score: {score}% - Category: {df['category'].iloc[idx]}")
        print(f"   Preview: {df['resume_text'].iloc[idx][:80]}...")
    
    # Step 5: Train Classification Models
    print("\n[STEP 5] Training Classification Models...")
    
    # Prepare data for classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['category'])
    
    print(f"\nClasses: {le.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    models = classifier.train_all_models(X_train, y_train)
    
    # Step 6: Evaluate Models
    print("\n[STEP 6] Evaluating Models...")
    
    results = {}
    for model_name, model in models.items():
        result = classifier.evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = result
    
    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1_score']
        }
        for name, res in results.items()
    }).T
    
    print(comparison_df)
    
    # Step 7: Cross-Validation
    print("\n[STEP 7] Cross-Validation...")
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        classifier.cross_validate(model, tfidf_matrix, y, cv=5)
    
    # Step 8: Feature Importance (Logistic Regression)
    print("\n[STEP 8] Feature Importance Analysis...")
    
    lr_model = models['Logistic Regression']
    feature_names = feature_extractor.get_feature_names()
    
    importance = classifier.get_feature_importance(lr_model, feature_names, top_n=10)
    
    if importance:
        print("\nTop Features per Class (Logistic Regression):")
        for class_name, features in importance.items():
            print(f"\n{class_name} ({le.classes_[int(class_name.split('_')[1])]}):")
            print("  Top Positive Features:")
            for feat, coef in features['top_positive'][:5]:
                print(f"    {feat}: {coef:.4f}")
    
    # Step 9: Test on New Resume
    print("\n[STEP 9] Testing on New Resume...")
    
    new_resume = "Experienced Python developer with expertise in Django, Flask, and PostgreSQL. Built RESTful APIs and microservices. Strong knowledge of software design patterns."
    
    print(f"\nNew Resume: {new_resume}")
    
    # Preprocess and vectorize
    new_processed = preprocessor.preprocess(new_resume)
    new_vector = feature_extractor.transform_tfidf([new_processed])
    
    # Classify with both models
    print("\nClassification Results:")
    for model_name, model in models.items():
        result = classifier.classify_resume(model, new_vector, class_names=le.classes_)
        print(f"\n{model_name}:")
        print(f"  Predicted Category: {result['predicted_class']}")
        if result['confidence']:
            print(f"  Confidence: {result['confidence']}%")
    
    # Match to job description
    match_score = feature_extractor.match_resume_to_job(new_vector, job_vector)
    print(f"\nMatch to Job Description: {match_score}%")
    
    # Step 10: Save Models
    print("\n[STEP 10] Saving Models...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save best model (highest accuracy)
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = models[best_model_name]
    
    classifier.save_model(best_model, f'models/best_classifier_{best_model_name.replace(" ", "_")}.pkl')
    
    # Save vectorizer
    import joblib
    joblib.dump(feature_extractor.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    print("\n" + "="*70)
    print("RESUME SCREENING SYSTEM - COMPLETE!")
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()