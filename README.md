
# 📄 Resume Classification Using NLP Techniques
This project showcases a modern Natural Language Processing (NLP) pipeline for classifying resumes into predefined job categories using **transformer-based deep learning models**, specifically **BERT (Bidirectional Encoder Representations from Transformers)**. It provides a scalable and intelligent solution to automate the initial screening phase in recruitment processes.

## 🚀 Project Significance
Manual resume screening can be time-consuming, inconsistent, and prone to bias. By leveraging the power of NLP and pretrained transformer models like BERT, this project automates the classification of resume content, improving efficiency, accuracy, and fairness in candidate shortlisting. It’s particularly useful for HR tech applications, recruitment platforms, and talent management systems.

## 📊 Key Components
- **PDF Text Extraction**: Extracts textual content from resumes using `pdfplumber`.
- **Text Preprocessing**: Cleans and tokenizes text using `nltk` with lowercasing, stopword removal, and optional stemming.
- **Exploratory Data Analysis (EDA)**: Visualizes word distributions and class imbalances using `seaborn` and `wordcloud`.
- **Modeling**:
  - ✅ **Fine-tuned BERT Transformer** from Hugging Face's `transformers` library
  - ❌ (Commented Out) Classical model: Naive Bayes
  - ❌ (Commented Out) Deep Learning model: LSTM
- **Training**:
  - Uses `Adam` optimizer with learning rate tuning and scheduling.
  - Includes early stopping and evaluation metrics like accuracy and classification report.
- **Imbalanced Dataset Handling**: Integrates oversampling techniques using `imbalanced-learn` to address class imbalance.

## 🛠️ Libraries & Tools
`transformers`, `datasets`, `torch`, `tensorflow`, `scikit-learn`, `pdfplumber`, `nltk`, `wordcloud`, `seaborn`, `imbalanced-learn`, etc.
