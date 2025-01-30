# Cybersecurity-Data-Breaches---Machine-Learning-Project
Modeling  using classification and clustering technique.
Cybersecurity Data Breaches - Machine Learning Project

📌 Project Overview

This project analyzes cybersecurity data breaches using Exploratory Data Analysis (EDA), Machine Learning Classification, and Clustering techniques. We achieved an impressive 92% accuracy for both classification and clustering models. Our approach includes data preprocessing, feature engineering, and model optimization to gain meaningful insights and build robust predictive models.

📊 Dataset
click [here](https://www.kaggle.com/datasets/gojoyuno/cyber-breach-analysis-dataset) for dataset

The dataset contains cybersecurity data breach records with features such as organization type, attack method, data sensitivity, and breach impact.

⚡ Key Steps & Techniques Used

1️⃣ Exploratory Data Analysis (EDA)

Data Cleaning: Handling missing values, removing duplicates, and correcting data types.

Outlier Detection & Removal: Using IQR (Interquartile Range) .

Skewness Removal & Normalization:

Applied log transformation .

Standardized numerical features to follow a normal distribution.

Feature Visualization:

Used Matplotlib, Seaborn for detailed insights.

Distribution plots, histograms, scatter plots, and correlation heatmaps.

2️⃣ Feature Engineering

Text Processing with TF-IDF:

Used TF-IDF vectorization for text-based features such as attack methods and organization descriptions.

Encoding Categorical Variables:

Used One-Hot Encoding and Label Encoding.

Feature Scaling:

StandardScaler for numerical variables.

3️⃣ Machine Learning Modeling

✅ Classification Model (92% Accuracy)

Target Variable: Binary classification (e.g., Sensitive or not)

Algorithm Used:

Random Forest Classifier


✅ Clustering Model (92% Accuracy)

K-Means Clustering:

Determined the optimal number of clusters using Elbow Method & Silhouette Score 92%.

Scaled data to improve cluster separation.

DBSCAN & Hierarchical Clustering:

Compared results with other clustering techniques.

4️⃣ Model Evaluation

Classification Metrics:

Accuracy, Precision, Recall, F1-Score, ROC-AUC Curve.

Clustering Metrics:

Silhouette Score .

5️⃣ Insights & Conclusions

Identified key factors influencing data breaches.

Showed how certain industries are more vulnerable to attacks.

Built a predictive model that helps in risk assessment.

🚀 Tools & Libraries Used

Python: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

Text Processing: NLTK, TF-IDF Vectorizer.

Machine Learning Models: Logistic Regression, Random Forest, SVM, K-Means.

Data Preprocessing: StandardScaler, Log Transformation, One-Hot Encoding.
