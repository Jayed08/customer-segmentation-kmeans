# Customer Segmentation using K-Means Clustering

An end-to-end unsupervised machine learning project that segments retail customers based on demographics, purchasing behavior, and marketing engagement. The project applies feature engineering, clustering analysis, customer profiling, and business recommendations, and is deployed as an interactive Streamlit dashboard.

---

## Live Demo

🔗 **Streamlit:** https://YOUR-STREAMLIT-URL.streamlit.app

## GitHub Repository

🔗 https://github.com/Jayed08/customer-segmentation-kmeans

---

# Problem Statement

Businesses often treat all customers similarly despite having very different purchasing behaviors. This leads to inefficient marketing campaigns, lower customer engagement, and unnecessary marketing costs.

The objective of this project is to automatically discover meaningful customer segments using unsupervised learning so that businesses can design personalized marketing strategies for different customer groups.

---

# Dataset

**Source:** Kaggle - Customer Personality Analysis

**Dataset Size**

- 2,240 customers
- Demographic information
- Purchase history
- Campaign responses
- Website activity
- Product category spending

---

# Project Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Feature Scaling
      │
      ▼
Optimal K Selection
      │
      ▼
K-Means Clustering
      │
      ▼
Cluster Profiling
      │
      ▼
Business Recommendations
      │
      ▼
Streamlit Deployment
```

---

# Feature Engineering

Several meaningful customer behavior features were created.

### Demographic Features

- Customer Age
- Customer Tenure
- Family Size
- Number of Children

### Spending Features

- Total Spending
- Product Spending Shares
  - Wines
  - Fruits
  - Meat
  - Fish
  - Sweets
  - Gold

### Purchase Behaviour

- Total Purchases
- Web Purchase Ratio
- Store Purchase Ratio
- Catalog Purchase Ratio

### Marketing Features

- Campaign Acceptance Count

---

# Data Preprocessing

- Missing value handling
- Date parsing
- Outlier removal using IQR
- Feature scaling using StandardScaler
- Correlation analysis

---

# Clustering Models

The following approaches were evaluated.

- K-Means Clustering
- Agglomerative Clustering

### Cluster Evaluation Metrics

- Elbow Method (WCSS)
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

K-Means with **K = 3** was selected as the final clustering model.

---

# Customer Segments

## Premium Customers

- High income
- Highest spending
- High purchase frequency
- Highly responsive to campaigns

### Business Strategy

- VIP rewards
- Premium offers
- Early product access

---

## Family Shoppers

- Moderate income
- Larger households
- Consistent purchasing behavior

### Business Strategy

- Family bundles
- Loyalty programs
- Personalized promotions

---

## Budget Shoppers

- Lower income
- Lower spending
- Lower campaign engagement

### Business Strategy

- Discount campaigns
- Seasonal offers
- Cost-efficient digital marketing

---

# Visualizations

The project includes multiple analytical visualizations.

- Correlation Heatmap
- Elbow Curve
- PCA Cluster Visualization
- Customer Segment Comparison
- Customer Segment Heatmap
- Agglomerative Dendrogram

---

# Streamlit Dashboard

The deployed dashboard includes

- Customer Segment Prediction
- Interactive Customer Input
- Segment Explorer
- Customer Profile Comparison
- Spending Breakdown
- Marketing Recommendations
- Dataset Statistics

---

# Tech Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- SciPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Deployment

- Streamlit
- Joblib

---

# Repository Structure

```
customer-segmentation-kmeans/
│
├── app.py
├── Customer_Segmentation.ipynb
├── customer_segmentation_model.joblib
├── requirements.txt
├── README.md
│
├── images/
│   ├── customer_segment_comparison.png
│   ├── customer_segments_pca.png
│   ├── customer_segment_profile_heatmap.png
│   ├── elbow_method.png
│   └── feature_correlation_heatmap.png
│
└── data/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Jayed08/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# Future Improvements

- Gaussian Mixture Models (GMM)
- Automatic cluster naming using LLMs
- Customer recommendation engine
- Interactive cluster editing
- Cloud deployment with monitoring

---

# Author

**Jayed Ansari**

- GitHub: https://github.com/Jayed08
- LinkedIn: YOUR_LINKEDIN
