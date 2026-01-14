# Machine Learning Prediction System 📊🤖

## Project Overview
This project applies **Machine Learning techniques** to analyze a real-world dataset and build predictive models.  
The main goal is to discover meaningful patterns in the data and compare different machine learning algorithms based on their performance.

This project was developed as part of an academic course and follows a full ML pipeline:
data preprocessing → modeling → evaluation → clustering → dimensionality reduction.



## Problem Definition
- **Machine Learning Type:** Supervised Learning  
- **Reason:** The dataset includes labeled data, and the goal is to predict a target variable.

### Target Variable
- The variable we aim to predict represents the final outcome of the problem.

### Input Features
- Numerical and categorical features that have a direct impact on the prediction.
- Feature selection was guided by data understanding and exploratory analysis.



## Dataset
- Publicly available dataset
- Contains both numerical and categorical attributes
- Includes missing values and required preprocessing



## Data Preprocessing & Feature Engineering
### Steps Performed:
- Handling missing values
- Encoding categorical features
- Scaling numerical features
- Outlier detection
- Exploratory Data Analysis (EDA)

### Why These Steps Are Important?
- Improve model accuracy
- Ensure fair comparison between algorithms
- Reduce noise and bias in the data



## Machine Learning Models Used
The following models were implemented using **scikit-learn**:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Neural Network (MLP)

### Training Strategy
- Dataset split: **80% Training / 20% Testing**
- Hyperparameter tuning using **Cross-Validation**



## Clustering & Dimensionality Reduction
- **K-Means Clustering** was used to identify hidden patterns in the data.
- **Principal Component Analysis (PCA)** was applied to:
  - Reduce dimensionality
  - Improve visualization
  - Reduce computational complexity



## Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Best Model
- The best-performing model was selected based on balanced performance across all metrics.



## Ethical Considerations
- Incorrect predictions may lead to poor decision-making.
- Bias in data can negatively affect fairness and model reliability.
- Responsible data usage and evaluation are critical when deploying ML systems.



## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn



## Author
- Shahad Almasri
- Lilian Alhalabi
**Shahed Almasri**  
Data Science / Computer Science Student  
