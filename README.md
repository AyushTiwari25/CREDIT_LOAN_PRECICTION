# 📊 **Credit (Loan) Default Prediction Project** 🏦

## 📝 **Overview**
This project utilizes **machine learning** to predict the likelihood of **loan default** using a dataset with key financial indicators. By leveraging a **logistic regression model**, we aim to provide accurate predictions, aiding lenders in making informed credit decisions. The project includes **data preprocessing**, **feature engineering**, **model training**, and **evaluation**.

---

## 📂 **Dataset Information**
The dataset consists of **2000 entries** with **5 features**, each representing financial or demographic information relevant to predicting default status.

| 📊 **Column**         | 📜 **Description**                                      |
|----------------------|---------------------------------------------------------|
| `Income`             | Annual income of the applicant                          |
| `Age`                | Age of the applicant                                    |
| `Loan`               | Loan amount applied for                                 |
| `Loan to Income`     | Loan amount divided by income (engineered feature)     |
| `Default`            | Target variable indicating loan default status (0: No Default, 1: Default) |

- **Rows**: 2000
- **Columns**: 5
- **Data Link**: [Credit Default Dataset](https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv)

---

## ⚙️ **Project Setup Instructions**

To get started with this project, follow these steps to clone the repository and install required packages.

### 1. Clone the Repository
To clone the repository, use the following command in your terminal:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_FOLDER_NAME>
```

### 2. Install Dependencies
Ensure you have the necessary libraries installed. This project primarily uses `pandas` and `scikit-learn`.

```bash
pip install pandas scikit-learn
```

### 3. Data Exploration and Preprocessing
The first step involves understanding the dataset through summary statistics and visualizations. You can load the dataset and view its structure with the following code:

```python
import pandas as pd

# Load Dataset
data_url = 'https://raw.githubusercontent.com/ybifoundation/Dataset/main/Credit%20Default.csv'
df = pd.read_csv(data_url)

# Check Data Shape
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
```

---

## 🚀 **Model Development and Training**

### 📈 **Data Preparation**
The features (`Income`, `Age`, `Loan`, `Loan to Income`) were extracted as `X`, while the target variable (`Default`) was labeled as `y`.

```python
# Define Features and Target Variable
X = df[['Income', 'Age', 'Loan', 'Loan to Income']]
y = df['Default']
```

### 🔀 **Train-Test Split**
The data was split into training (70%) and test (30%) sets to train and validate the model effectively.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)
print("Training Set:", X_train.shape)
print("Test Set:", X_test.shape)
```

### 🤖 **Model Selection**
After testing multiple models, logistic regression was chosen for its balance of simplicity and accuracy.

```python
from sklearn.linear_model import LogisticRegression

# Initialize and Train Model
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 🧪 **Model Evaluation**
The model was evaluated on the test set, achieving **95% accuracy**. Below is the evaluation code and results.

```python
from sklearn.metrics import accuracy_score, classification_report

# Predict on Test Data
y_pred = model.predict(X_test)

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)  # Output: 0.95

# Classification Report
print(classification_report(y_test, y_pred))
```

#### **Performance Metrics**
| 🏷️ **Metric**      | 📊 **Class 0 (No Default)** | 📉 **Class 1 (Default)** |
|--------------------|-----------------------------|--------------------------|
| **Precision**      | 0.97                        | 0.83                     |
| **Recall**         | 0.97                        | 0.79                     |
| **F1-Score**       | 0.97                        | 0.81                     |

- **Best Model**: Logistic Regression
- **Accuracy**: 95%
- **R-squared value**: Not directly applicable for classification models, but high accuracy indicates reliable predictive performance.

---

## 📊 **Exploratory Data Analysis (EDA)**

During EDA, the following insights were gathered:
- **Income** and **Loan Amount** are key factors influencing default probability.
- Most applicants are under the age of 40.
- High-income applicants tend to have lower default rates.

---

## 💻 **Usage Instructions**

You can use the trained model for predicting loan default on new applicant data. Here’s how:

### **Predict Loan Default**

```python
# Predict on New Data
new_data = [[60000, 35, 5000, 0.08]]
default_prediction = model.predict(new_data)
print("Loan Default Prediction:", "Default" if default_prediction[0] == 1 else "No Default")
```

---

## 🏆 **Key Takeaways**
- This **Credit (Loan) Default Prediction** model provides reliable predictions with an accuracy of 95%.
- Logistic regression proved effective in capturing the relationship between financial indicators and default risk.
- This project highlights the importance of **feature engineering** (e.g., Loan to Income ratio) in improving model performance.

---

## 📬 **Contact and Contributions**

If you find this project helpful or have suggestions, feel free to **open an issue** or **submit a pull request**. Let's collaborate to enhance this tool and make credit risk prediction accessible! 🙌✨

---
