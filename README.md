# Sentiment-Analysis-of-Amazon-Review-Using-Bert-DistilBert
This project applies advanced transformer-based models (BERT and Distil-BERT) to analyze 24,948 Amazon product reviews. The goal is to classify them into positive, neutral, and negative sentiments to support brand reputation tracking and crisis management.
## Key Results:
BERT Accuracy: 92.75%

Distil-BERT Accuracy: 95.20% (60% faster than BERT)

Distil-BERT is more suitable for real-time monitoring

BERT offers deeper contextual understanding, ideal for complex linguistic cases
## Problem Statement
With the rapid growth of e-commerce, analyzing customer reviews is crucial for understanding customer sentiment, detecting issues, and improving product offerings. Traditional models fail to handle context-rich or mixed-emotion sentences. Thus, transformer-based NLP models are applied.
##  Dataset
Source: Kaggle Amazon Product Reviews -<a herf="https://www.kaggle.com/datasets/gunjalakshmanarao/comprehensive-product-reviews-with-ratings/data">

Total Records: 25,000

Split:

Training: 20,000

Validation: 2,494

Testing: 2,454
## Preprocessing Steps
Noise removal (HTML, symbols, stop words)

Lowercasing

Tokenization (Distil-BERT Tokenizer)

Lemmatization & Stemming

Sentiment Labeling:

⭐ 4-5 stars: Positive

⭐ 3 stars: Neutral

⭐ 1-2 stars: Negative

## Models Used
1. BERT (Bidirectional Encoder Representations from Transformers)
Deep contextual understanding

Bidirectional transformer layers

High precision in complex sentiment analysis

2. Distil-BERT
60% faster

Lightweight version of BERT with 97% of its performance

Ideal for real-time and resource-constrained systems
## Model Training
Optimizer: AdamW

Learning Rate: 2e-5

Epochs:

BERT: 5

Distil-BERT: 40

Batch Size:

BERT: 32

Distil-BERT: 128

Loss Function: Cross-Entropy
## Visualizations
Confusion Matrices -<a herf="">

F1-Score, Precision, Recall Charts

Accuracy Comparison Graphs

## Conclusion
Distil-BERT is ideal for fast, real-time analysis.

BERT excels in understanding complex sentence structures.

Both models significantly outperform traditional ML methods.

This pipeline helps brands track sentiment trends, identify issues early, and boost customer satisfaction.

## Future Work
Apply to multilingual datasets

Integrate aspect-based sentiment analysis (ABSA)

Combine models for hybrid approaches

Deploy as a web API for real-time sentiment monitoring

## Requirements
transformers
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow or pytorch
## How to Run
-<a herf=">git clone https://github.com/yourusername/sentiment-amazon-bert.git
cd sentiment-amazon-bert">
## Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt

## Run training or inference scripts:

bash
Copy
Edit
python train_bert.py
python train_distilbert.py



