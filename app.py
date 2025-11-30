import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from resume_parser import ResumeParser

st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📄 AI Resume Screening System</h1>', unsafe_allow_html=True)
st.markdown("**Match resumes to job descriptions using NLP and Machine Learning**")

# Initialize components
@st.cache_resource
def load_models():
    try:
        classifier = joblib.load('models/best_classifier_Logistic_Regression.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        return classifier, vectorizer, label_encoder
    except:
        return None, None, None

classifier, vectorizer, label_encoder = load_models()
preprocessor = TextPreprocessor(use_lemmatization=True)
parser = ResumeParser()

# Sidebar
st.sidebar.header("⚙️ Options")
mode = st.sidebar.radio("Select Mode", ["Resume-Job Matching", "Resume Classification", "Bulk Screening"])

if mode == "Resume-Job Matching":
    st.header("🎯 Resume-Job Matching")
    st.markdown("Find the best match between a resume and job description using cosine similarity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Enter job description",
            height=300,
            placeholder="e.g., Looking for a Python developer with 3+ years experience in machine learning..."
        )
    
    with col2:
        st.subheader("📄 Resume")
        resume_input_method = st.radio("Input method", ["Paste Text", "Upload File"])
        
        if resume_input_method == "Paste Text":
            resume_text = st.text_area(
                "Paste resume text",
                height=300,
                placeholder="Paste the resume content here..."
            )
        else:
            uploaded_file = st.file_uploader("Upload resume (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])
            resume_text = ""
            if uploaded_file:
                # Save temporarily and parse
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                resume_text = parser.parse_resume(temp_path)
                os.remove(temp_path)
                st.success(f"✅ Parsed {uploaded_file.name}")
    
    if st.button("🔍 Calculate Match Score", type="primary"):
        if job_description and resume_text:
            with st.spinner("Analyzing..."):
                # Preprocess
                job_processed = preprocessor.preprocess(job_description)
                resume_processed = preprocessor.preprocess(resume_text)
                
                # Create temporary vectorizer if models not loaded
                if vectorizer is None:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    temp_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
                    vectors = temp_vectorizer.fit_transform([job_processed, resume_processed])
                    job_vector = vectors[0]
                    resume_vector = vectors[1]
                else:
                    job_vector = vectorizer.transform([job_processed])
                    resume_vector = vectorizer.transform([resume_processed])
                
                # Calculate similarity
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(resume_vector, job_vector)[0][0]
                match_score = similarity * 100
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Match Score", f"{match_score:.1f}%")
                
                with col2:
                    if match_score >= 70:
                        st.success("✅ Strong Match")
                    elif match_score >= 50:
                        st.warning("⚠️ Moderate Match")
                    else:
                        st.error("❌ Weak Match")
                
                with col3:
                    st.metric("Similarity", f"{similarity:.3f}")
                
                # Progress bar
                st.progress(match_score / 100)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if match_score >= 70:
                    st.success("**Highly Recommended** - This candidate is an excellent match for the position!")
                elif match_score >= 50:
                    st.info("**Consider for Interview** - Good match with some gaps. Review specific requirements.")
                else:
                    st.warning("**Not Recommended** - Significant gaps between resume and job requirements.")
        else:
            st.error("Please provide both job description and resume!")

elif mode == "Resume Classification":
    st.header("🏷️ Resume Classification")
    st.markdown("Classify resumes into categories using ML models")
    
    if classifier is None:
        st.error("⚠️ Models not found! Please train the models first by running `python main.py`")
    else:
        resume_input = st.text_area(
            "Enter resume text",
            height=300,
            placeholder="Paste resume content here..."
        )
        
        uploaded_file = st.file_uploader("Or upload resume file", type=['pdf', 'docx', 'txt'])
        
        if uploaded_file:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            resume_input = parser.parse_resume(temp_path)
            os.remove(temp_path)
            st.success(f"✅ Parsed {uploaded_file.name}")
        
        if st.button("🔮 Classify Resume", type="primary"):
            if resume_input:
                with st.spinner("Classifying..."):
                    # Preprocess
                    processed = preprocessor.preprocess(resume_input)
                    
                    # Vectorize
                    vector = vectorizer.transform([processed])
                    
                    # Predict
                    prediction = classifier.predict(vector)[0]
                    probabilities = classifier.predict_proba(vector)[0]
                    
                    # Get category name
                    category = label_encoder.inverse_transform([prediction])[0]
                    confidence = max(probabilities) * 100
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Classification Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Category", category)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Show all probabilities
                    st.subheader("📈 Category Probabilities")
                    
                    prob_df = pd.DataFrame({
                        'Category': label_encoder.classes_,
                        'Probability': probabilities * 100
                    }).sort_values('Probability', ascending=False)
                    
                    st.bar_chart(prob_df.set_index('Category'))
                    
                    # Extract skills
                    st.subheader("🔍 Extracted Information")
                    
                    # Contact info
                    contact = parser.extract_contact_info(resume_input)
                    if any(contact.values()):
                        st.write("**Contact Information:**")
                        if contact['email']:
                            st.write(f"📧 Email: {contact['email']}")
                        if contact['phone']:
                            st.write(f"📱 Phone: {contact['phone']}")
                        if contact['linkedin']:
                            st.write(f"💼 LinkedIn: {contact['linkedin']}")
            else:
                st.error("Please provide resume text!")

else:  # Bulk Screening
    st.header("📚 Bulk Resume Screening")
    st.markdown("Screen multiple resumes against a job description")
    
    job_description = st.text_area(
        "Job Description",
        height=200,
        placeholder="Enter the job description..."
    )
    
    uploaded_files = st.file_uploader(
        "Upload multiple resumes",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if st.button("🚀 Screen All Resumes", type="primary"):
        if job_description and uploaded_files:
            with st.spinner(f"Screening {len(uploaded_files)} resumes..."):
                results = []
                
                # Process job description
                job_processed = preprocessor.preprocess(job_description)
                
                # Create vectorizer if not loaded
                if vectorizer is None:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    temp_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
                    all_texts = [job_processed]
                    
                    # Parse all resumes first
                    resume_texts = []
                    for file in uploaded_files:
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, 'wb') as f:
                            f.write(file.getbuffer())
                        text = parser.parse_resume(temp_path)
                        os.remove(temp_path)
                        resume_texts.append(text)
                        all_texts.append(preprocessor.preprocess(text))
                    
                    # Fit vectorizer
                    vectors = temp_vectorizer.fit_transform(all_texts)
                    job_vector = vectors[0]
                    resume_vectors = vectors[1:]
                else:
                    job_vector = vectorizer.transform([job_processed])
                    resume_texts = []
                    resume_vectors_list = []
                    
                    for file in uploaded_files:
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, 'wb') as f:
                            f.write(file.getbuffer())
                        text = parser.parse_resume(temp_path)
                        os.remove(temp_path)
                        resume_texts.append(text)
                        processed = preprocessor.preprocess(text)
                        resume_vectors_list.append(vectorizer.transform([processed]))
                
                # Calculate similarities
                from sklearn.metrics.pairwise import cosine_similarity
                
                for i, file in enumerate(uploaded_files):
                    if vectorizer is None:
                        resume_vector = resume_vectors[i]
                    else:
                        resume_vector = resume_vectors_list[i]
                    
                    similarity = cosine_similarity(resume_vector, job_vector)[0][0]
                    match_score = similarity * 100
                    
                    # Extract contact
                    contact = parser.extract_contact_info(resume_texts[i])
                    
                    results.append({
                        'Filename': file.name,
                        'Match Score': round(match_score, 2),
                        'Email': contact.get('email', 'N/A'),
                        'Phone': contact.get('phone', 'N/A'),
                        'Status': '✅ Strong' if match_score >= 70 else '⚠️ Moderate' if match_score >= 50 else '❌ Weak'
                    })
                
                # Sort by match score
                results_df = pd.DataFrame(results).sort_values('Match Score', ascending=False)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Screening Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Resumes", len(results_df))
                with col2:
                    strong_matches = len(results_df[results_df['Match Score'] >= 70])
                    st.metric("Strong Matches", strong_matches)
                with col3:
                    avg_score = results_df['Match Score'].mean()
                    st.metric("Average Score", f"{avg_score:.1f}%")
                
                # Show table
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "screening_results.csv",
                    "text/csv"
                )
        else:
            st.error("Please provide job description and upload resumes!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & NLP | [GitHub](https://github.com/Akrati36/resume-screening-system)")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    ### Resume-Job Matching
    1. Enter a job description
    2. Paste resume text or upload a file
    3. Click "Calculate Match Score"
    4. View similarity score and recommendations
    
    ### Resume Classification
    1. Train models first: `python main.py`
    2. Paste resume text or upload file
    3. Click "Classify Resume"
    4. View predicted category and confidence
    
    ### Bulk Screening
    1. Enter job description
    2. Upload multiple resume files
    3. Click "Screen All Resumes"
    4. View ranked results and download CSV
    """)