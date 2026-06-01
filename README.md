# Intelligent Data Assistant for MS Excel

> "Transforming Data into Intelligent Insights"

A fully interactive, AI-powered data science web application built with Streamlit. 
Designed as a universal data analysis and machine learning pipeline, it allows anyone 
— regardless of technical background — to upload a dataset and instantly explore, 
clean, visualize, and model their data through a simple command-driven interface.

This project was developed as a Final Year Project at the University of Haripur, 
Department of Information Technology, in 2026.

---

## Overview

Most people working with data in Excel or CSV files do not have the technical 
knowledge to run machine learning models or write Python scripts. This application 
bridges that gap entirely. A user uploads their dataset, selects from 21 built-in 
commands organized across five intelligent modules, and the system handles 
everything — from cleaning missing values to training a predictive model and 
generating automated insights — all through a clean, professional web interface.

---

## Features

### Module 1 — Smart Exploration
- Preview the top rows of any uploaded dataset
- Generate full statistical summaries
- Inspect column data types and structure
- Calculate means, totals, and min/max values across all numeric features

### Module 2 — Automated Data Cleaning
- Smart missing value handling using median imputation
- Duplicate record detection and removal
- Text standardization to title case across categorical columns
- Z-score based outlier detection
- Full feature normalization using Standard Gaussian scaling

### Module 3 — Visual Intelligence
- Correlation heatmap across numeric features
- Interactive bar charts of feature averages
- Box plots for distribution analysis
- Pie charts for categorical breakdowns
- Line charts for trend visualization

### Module 4 — Advanced AI and NLP
- AI Semantic Clustering using Sentence Transformers to detect hidden 
  patterns in text columns
- t-SNE visualization of AI-generated clusters in 2D space
- Automated Insight Narrative Engine that summarizes key statistical 
  and categorical patterns in plain language

### Module 5 — Predictive Machine Learning
- One-click Random Forest model training on any numeric target variable
- Feature importance analysis showing which columns drive predictions most
- Custom Accuracy Builder Lab where users manually select features and 
  target variables to train and evaluate their own models

---

## System Workflow

Upload (Excel / CSV) → Process & Clean → Analyze Patterns → 
Visualize Data → Predict (ML) → Insights & Reports

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Web application framework |
| Pandas & NumPy | Data manipulation and numerical processing |
| Scikit-learn | Machine learning models and preprocessing |
| Matplotlib & Seaborn | Data visualization |
| Sentence Transformers | NLP-based semantic clustering |
| KMeans & t-SNE | Unsupervised clustering and dimensionality reduction |

---

## How to Run

1. Clone the repository:
