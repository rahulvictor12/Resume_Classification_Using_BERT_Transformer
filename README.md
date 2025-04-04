
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

## Why Switch from Naive Bayes and LSTM to Transformers (BERT)?

In this project, classical machine learning (Naive Bayes) and sequential deep learning models (LSTM) were initially explored. However, both were later commented out in favor of using **Transformer-based models**, specifically **BERT**. Here's why this change was made:

### 1. Performance & Accuracy
- **Naive Bayes** relies on the assumption of feature independence, which does not hold in natural language where word meaning is context-dependent.
- **LSTM** improves upon this by handling sequential data but struggles with long-range dependencies and requires extensive training and tuning.
- **BERT**, by contrast:
  - Uses self-attention to capture global dependencies in text.
  - Is pretrained on large corpora and fine-tunable, providing high performance on text classification tasks.

### 2. Semantic Understanding
- Resume classification requires nuanced language understanding to differentiate between roles and industries.
- BERT’s bidirectional context capturing allows for a much deeper and accurate comprehension of resume content than NB or LSTM.

### 3. Transfer Learning Advantage
- BERT is already trained on vast datasets and requires only fine-tuning.
- This is especially advantageous when working with limited labeled resume data.

### 4. Industry Relevance
- Transformer models like BERT are now the gold standard in NLP tasks.
- Adopting BERT makes the project more aligned with current best practices and expectations in the field.

**In summary**, BERT was adopted to leverage its superior contextual understanding, pretrained knowledge, and state-of-the-art performance for more robust and reliable resume classification.

## Observations:
1. There were no Null Values so there was no need for an imputation or Null Value Handling Required
2. There was an categorical imbalance of data which was the reason behind the use of a RandomOverSampler (Could have used SMOTE but smote requires Continous Sample Data and we would have to generate the embeddings first and then synthesize the new samples and reintegrate them into BERT which feels like a stretch)
