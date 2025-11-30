# 🚀 Quick Start Guide - Resume Screening System

Get your Resume Screening System running in 5 minutes!

## Step 1: Clone the Repository

```bash
git clone https://github.com/Akrati36/resume-screening-system.git
cd resume-screening-system
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Or run in Python:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Step 4: Train the Models

```bash
python main.py
```

**Expected Output:**
```
======================================================================
RESUME SCREENING SYSTEM - NLP & ML
======================================================================

[STEP 1] Loading Resume Data...
Loaded 10 resumes

[STEP 2] Preprocessing Resume Text...
- Tokenization
- Stopword removal
- Lemmatization

[STEP 3] TF-IDF Vectorization...
TF-IDF Matrix Shape: (10, 500)

[STEP 5] Training Classification Models...
=== Training Logistic Regression ===
=== Training Naive Bayes ===

[STEP 6] Evaluating Models...
Accuracy:  0.9000

BEST MODEL: Logistic Regression
Accuracy: 0.9000
======================================================================
```

**Runtime**: ~30 seconds

## Step 5: Launch Web App

```bash
streamlit run app.py
```

Open browser to: `http://localhost:8501`

## 🎯 Quick Test

### Test 1: Resume-Job Matching

**Job Description:**
```
Looking for a Python developer with 3+ years experience in machine learning. 
Must know TensorFlow, scikit-learn, and have strong NLP skills.
```

**Resume:**
```
Python developer with 5 years experience in ML and data science. 
Expert in TensorFlow, PyTorch, scikit-learn. Built NLP models and 
recommendation systems.
```

**Expected Match Score**: 70-85%

### Test 2: Resume Classification

**Resume:**
```
Full stack developer with React, Node.js, MongoDB experience. 
Built scalable web applications. 4 years in software development.
```

**Expected Category**: Software Engineering
**Confidence**: 85-95%

## 📁 Using Your Own Data

### Option 1: Use Sample Data (Default)
The system comes with sample resumes built-in. Just run `python main.py`

### Option 2: Use Your Resume Dataset

Create a CSV file: `data/resume_dataset.csv`

**Required columns:**
- `resume_text` - Full resume text
- `category` - Job category (e.g., "Data Science", "Software Engineering")

**Example:**
```csv
resume_text,category
"Python developer with ML experience...",Data Science
"Java backend engineer with Spring Boot...",Software Engineering
```

Update `main.py`:
```python
def load_sample_data():
    df = pd.read_csv('data/resume_dataset.csv')
    return df
```

### Option 3: Parse Resume Files

Place resume files in `data/resumes/` folder:
```
data/resumes/
├── john_doe_resume.pdf
├── jane_smith_resume.docx
└── bob_johnson_resume.txt
```

Use the parser:
```python
from src.resume_parser import ResumeParser

parser = ResumeParser()
resumes = parser.parse_multiple_resumes('data/resumes/')
```

## 🎨 Web App Features

### 1. Resume-Job Matching
- Paste job description
- Upload resume (PDF/DOCX/TXT)
- Get match score (0-100%)
- View recommendations

### 2. Resume Classification
- Upload resume
- Get predicted category
- View confidence scores
- Extract contact info

### 3. Bulk Screening
- Upload multiple resumes
- Rank by match score
- Download CSV results

## 🔧 Common Issues

### Issue: "No module named 'nltk'"
```bash
pip install nltk
```

### Issue: "Resource punkt not found"
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: "Models not found"
Run training first:
```bash
python main.py
```

### Issue: "Cannot parse PDF"
Install PyPDF2:
```bash
pip install PyPDF2
```

## 📊 Understanding Results

### Match Scores
- **70-100%**: Strong match - Highly recommended
- **50-69%**: Moderate match - Consider for interview
- **0-49%**: Weak match - Not recommended

### Classification Confidence
- **80-100%**: Very confident
- **60-79%**: Confident
- **40-59%**: Uncertain
- **0-39%**: Low confidence

## 🎓 Next Steps

1. **Customize Categories**: Edit categories in `main.py`
2. **Add Skills**: Update skill keywords in `text_preprocessing.py`
3. **Tune Models**: Adjust parameters in `resume_classifier.py`
4. **Deploy**: Use Streamlit Cloud, Heroku, or Railway

## 💡 Pro Tips

- Use real resume data for better accuracy
- Train on at least 50-100 resumes per category
- Update stopwords for domain-specific terms
- Experiment with different TF-IDF parameters
- Use cross-validation to avoid overfitting

## 📚 Learn More

- [Full Documentation](README.md)
- [NLP Preprocessing Guide](src/text_preprocessing.py)
- [TF-IDF Explained](src/feature_extraction.py)
- [Classification Models](src/resume_classifier.py)

---

**Ready to screen resumes like a pro! 🎉**

Questions? Open an issue on GitHub!