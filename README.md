#### 🇹🇷 Turkish E-Commerce Sentiment Analysis with BERT
### An End-to-End Customer Insights Dashboard
## Project Overview

This project presents an end-to-end AI-powered sentiment analysis system that analyzes Turkish customer reviews collected from one of Türkiye’s leading e-commerce platforms (Hepsiburada).
By leveraging Natural Language Processing (NLP) and Transformer-based deep learning models, the system extracts actionable customer insights and presents them through an interactive Streamlit dashboard.

Using the BERTurk model, the project accurately classifies customer sentiment while overcoming the linguistic complexity of the Turkish language. The results are visualized in a business-friendly format, enabling companies to monitor customer satisfaction and identify improvement areas efficiently.

## Problem Solved:
Manually analyzing thousands of customer reviews is inefficient and error-prone. This system automates sentiment analysis, allowing businesses to track customer satisfaction in real time and make data-driven decisions faster.

## Project Demo

Watch the full demo showcasing:

Live sentiment prediction

Interactive analytics dashboard

Key customer insights
https://github.com/user-attachments/assets/1ec9dcf5-31d8-4d0c-9af6-c2cf18a22866
*If the video does not play automatically, please open the file “Video Project.mp4” from the repository assets.*

## Project Objectives & Key Achievements
# Data Engineering

Cleaned and processed 300,000+ raw customer reviews

Fixed noisy labels, duplicates, and missing values

Created a balanced, high-quality dataset of 15,000 samples using oversampling techniques

# Advanced NLP Modeling

Fine-tuned the BERTurk Transformer model, optimized for Turkish language structure

Achieved 97.90% classification accuracy

# Comparative Evaluation

Built a traditional baseline model (TF-IDF + Logistic Regression)

Demonstrated that BERTurk reduces the error rate by more than 5×

Validated improvements using scientific evaluation metrics

# Deployment & Productization

Deployed the trained model into an interactive Streamlit dashboard

Enabled non-technical users to explore insights and perform live predictions

# Technologies Used

Python – Core development language

HuggingFace Transformers – BERTurk model loading and fine-tuning

Streamlit – Interactive dashboard & web interface

Pandas – Data preprocessing and manipulation

Scikit-learn – Baseline model & evaluation metrics

Altair & Matplotlib / Seaborn – Data visualization

Google Colab – GPU-accelerated model training

## Project Workflow & Findings
1️⃣ Data Cleaning & Preparation

Cleaned 300K+ raw Hepsiburada reviews

Removed duplicates, missing entries, and labeling inconsistencies

Balanced rating distribution (1–5 stars) using oversampling

Normalized text for Transformer-based modeling

2️⃣ Modeling & Training

Baseline Model:

TF-IDF + Logistic Regression

Accuracy: 92.49%

Main Model:

BERTurk fine-tuned for 3 epochs

Accuracy: 97.90%

## Key Insight:
BERTurk successfully captured contextual sentiment in complex expressions such as:

“The product is good, but the delivery was terrible.”

3️⃣ Interactive Dashboard

The Streamlit dashboard includes:

Live Analysis: Real-time sentiment prediction for new reviews

General Insights:

Satisfaction levels

Complaint ratios

Sentiment distribution

Actionable Tables:

Filtered negative & neutral reviews to highlight improvement areas

# Installation & Usage Guide
# Important Note (Model Files)

Due to GitHub’s file size limitations, the fine-tuned BERTurk model files (~450 MB) are not included in this repository.

You can:

Review the full functionality via the demo video

Train the model yourself or provide the my_sentiment_model directory manually to run locally

🔧 Local Setup
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/Turkish-Ecommerce-Sentiment-Analysis.git
cd Turkish-Ecommerce-Sentiment-Analysis

2. Install Dependencies
pip install -r requirements.txt

3. Run the Application
streamlit run app.py
