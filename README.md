# Resume Ranking System

## Overview

The Resume Ranking System is an AI-powered tool that uses natural language processing and machine learning techniques to analyze job descriptions and rank resumes based on their relevance. The system provides a multi-dimensional analysis of candidate fit, helping recruiters and hiring managers efficiently identify the most qualified candidates.

## Key Features

- **Semantic Understanding**: Leverages BERT transformers to understand the meaning behind job descriptions and resumes beyond simple keyword matching
- **Multi-Dimensional Scoring**: Evaluates candidates across five components:
  - Semantic similarity
  - Required skills match
  - Preferred skills match
  - Experience match
  - Education match
- **Domain-Specific Analysis**: Categorizes skills across multiple professional domains
- **Adaptive Weighting**: Automatically adjusts scoring weights based on job type (technical, management, entry-level)
- **Detailed Insights**: Provides visual breakdown of matched and missing skills
- **Batch Processing**: Support for individual resume files or datasets with multiple resumes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-ranking-system.git
cd resume-ranking-system

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Using the Web Interface

1. **Starting the Server**:
   ```bash
   python resume_ranker.py --web --port 5000
   ```

2. **Accessing the Interface**:
   - Open your browser and navigate to http://localhost:5000

3. **Ranking Individual Resumes**:
   - Enter or paste the job description in the text area
   - Click "Browse" in the Resume Files section and select PDF, DOCX, or TXT files
   - Click "Rank Resumes" to start the ranking process
   - View the results displayed in a ranked table

4. **Using Dataset Mode** (for multiple resumes):
   - Enter the job description as above
   - In the "OR Upload Resume Dataset
