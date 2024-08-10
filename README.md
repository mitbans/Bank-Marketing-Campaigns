# Bank Direct Marketing Campaigns Analysis
## Predicting Long-Term Deposit Subscription Success

## Overview

This project aims to compare the performance of various classification algorithms to predict whether a client will subscribe to a long-term deposit based on data from multiple marketing campaigns conducted by a Portuguese banking institution. The classifiers compared in this project include K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM).

## Table of Contents

1. [Business Understanding](#business-understanding)
2. [Data Understanding](#data-understanding)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Deployment](#deployment)
7. [Usage](#usage)
8. [Repository Structure](#repository-structure)
9. [Notebook](#notebook)

## Business Understanding

The primary objective of this project is to optimize the direct marketing efforts of the bank by accurately predicting whether a client will subscribe to a long-term deposit. By achieving this, the bank can improve the efficiency of its marketing campaigns, reduce costs, and enhance customer targeting, leading to higher profitability and better resource allocation.

## Data Understanding

### Dataset Overview

- **Source:** UCI Machine Learning Repository, <a href="https://archive.ics.uci.edu/dataset/222/bank+marketing">link</a>
- **Content:** The dataset 'bank-additional-full.csv' comprises data collected from 17 marketing campaigns conducted between May 2008 and November 2010, resulting in 41,188 records and 20 inputs.
- **Attributes:** The dataset includes various features related to client demographics, previous campaign interactions, and economic indicators.
  
### Key Features:

- **Client Data:**
  - `age`: Age of the client
  - `job`: Job type (categorical)
  - `marital`: Marital status (categorical)
  - `education`: Education level (categorical)
  - `default`: Has credit in default? (categorical)
  - `housing`: Has housing loan? (categorical)
  - `loan`: Has personal loan? (categorical)
  
- **Contact Data:**
  - `contact`: Communication type (categorical)
  - `month`: Last contact month (categorical)
  - `day_of_week`: Last contact day of the week (categorical)
  - `duration`: Last contact duration in seconds (numeric)
  
- **Campaign Data:**
  - `campaign`: Number of contacts performed during this campaign (numeric)
  - `pdays`: Number of days since the client was last contacted (numeric)
  - `previous`: Number of contacts before this campaign (numeric)
  - `poutcome`: Outcome of the previous campaign (categorical)
  
- **Economic Indicators:**
  - `emp.var.rate`: Employment variation rate (numeric)
  - `cons.price.idx`: Consumer price index (numeric)
  - `cons.conf.idx`: Consumer confidence index (numeric)
  - `euribor3m`: Euribor 3 month rate (numeric)
  - `nr.employed`: Number of employees (numeric)

- **Target Variable:**
  - `y`: Subscription to a term deposit (binary: 'yes' or 'no')

### Potential Issues:

- `unknown` values in categorical variables such as `job`, `education`, `default`, `housing`, and `loan` represent missing data.
- `999` in `pdays` indicates that the client was not previously contacted.

## Data Preparation

### Steps Taken:

1. **Handling Missing Values:**
   - No missing values. 

3. **Data Cleaning:**
   - Removed 12 duplicate rows from the dataset.

4. **Correlation Analysis:**
    - A correlation matrix was calculated for the numerical features to understand the relationships between variables. Some key insights include:
      - `pdays` and `previous` show a strong negative correlation (-0.59).
      - `emp.var.rate`, `euribor3m`, and `nr.employed` are strongly positively correlated with each other.
      - `duration` has a minimal correlation with other features but is crucial in predicting the target variable.

4. **Splitting Data:**
    - The dataset was split into training and test sets to evaluate model performance.

5. **Feature Engineering: Preprocessing Pipeline**
  - **For Logistic Regression, KNN, and SVM**:
    - Numerical features were scaled using `StandardScaler`.
    - Categorical features were encoded using `OrdinalEncoder`.
  - **For Decision Trees**:
    - No scaling was required as decision trees are insensitive to feature scaling.
    - Categorical features were encoded using `OrdinalEncoder`.

## Model Training

### Models Compared:
  - **K-Nearest Neighbors (KNN)**
  - **Logistic Regression**
  - **Decision Trees**
  - **Support Vector Machines (SVM)**

### Initial Model Comparisons
Several machine learning models were trained, and their performance was compared:

## Model Comparison

| Model                  | Train Time | Train Accuracy | Test Accuracy | Precision | Recall  | F1-Score | AUC    |
|------------------------|------------|----------------|---------------|-----------|---------|----------|--------|
| Logistic Regression     | 0.24s      | 0.91           | 0.91          | 0.66      | 0.42    | 0.51     | 0.93   |
| Decision Tree           | 0.20s      | 1.00           | 0.89          | 0.51      | 0.53    | 0.52     | 0.73   |
| KNN                     | 0.0025s    | 0.93           | 0.90          | 0.59      | 0.40    | 0.47     | 0.86   |
| SVM                     | 188s       | 0.91           | 0.91          | 0.68      | 0.34    | 0.45     | 0.93   |

  - **Decision Tree** had the best performance with a comparable AUC of 0.93 and the highest F1 score of 0.57, indicating a good balance between precision and recall.
  - **Logistic Regression** also performed well, with a slightly lower F1 score but higher interpretability.

### Model Optimization
Hyperparameter tuning was performed using cross-validation and Grid Search:
  - **Logistic Regression**: Best parameters found were `C=0.1`, `penalty='l1'`, and `solver='saga'`.
  - **Decision Tree**: Best parameters were `criterion='gini'`, `max_depth=5`, `min_samples_leaf=1`, and `min_samples_split=2`.
  - **KNN**: Optimal parameters were `n_neighbors=21`, `p=2`, and `weights='distance'`.
  - **SVM**: The RBF kernel was found to be the best, but the model was computationally expensive for the given dataset.

## Evaluation

### Metrics Used:
  - **Accuracy**: The proportion of correctly classified instances.
  - **Precision, Recall, F1-Score**: To assess the balance between false positives and false negatives.
  - **ROC-AUC**: To evaluate the model's ability to distinguish between the two classes.

### Model Performance Summary
The following table summarizes the performance metrics for the improved models:

| Model                | Train Time (s) | Train Accuracy | Test Accuracy | Precision | Recall  | F1 Score | AUC     |
|----------------------|----------------|----------------|---------------|-----------|---------|----------|---------|
| Logistic Regression  | 95.69          | 0.9107         | 0.9098        | 0.6576    | 0.4159  | 0.5096   | 0.9333  |
| Decision Trees       | 98.78          | 0.9175         | 0.9143        | 0.6512    | 0.5151  | 0.5752   | 0.9304  |
| KNN                  | 839.37         | 1.0000         | 0.9024        | 0.6330    | 0.3179  | 0.4232   | 0.9129  |

### Key Findings

1. **Logistic Regression**:
   - Achieved a balanced performance with a strong AUC of 0.9333, indicating good model discrimination between classes.
   - Precision and recall are moderate, with room for improvement in the recall to capture more true positives.
     
2. **Decision Trees**:
   - The Decision Tree model delivered the best F1 Score of 0.5752, balancing precision and recall effectively.
   - This model also showed robust performance across all metrics, with the highest test accuracy of 0.9143.

3. **K-Nearest Neighbors (KNN)**:
   - While KNN achieved perfect train accuracy, indicating it might be overfitting, its test accuracy was slightly lower at 0.9024.
   - KNN has the lowest recall at 0.3179, indicating it missed many true positives, but it maintained a decent AUC.

### Conclusion: Best Model

The Decision Tree model stands out as the best performer for this classification task, especially in terms of the F1 score and test accuracy. This model offers a good balance between precision and recall, making it a strong candidate for deployment. Logistic Regression also performed well and could be considered depending on the specific use case requirements. However, KNN, despite its high training accuracy, may require further tuning or regularization to improve generalization and recall.

These findings suggest that feature engineering, further hyperparameter tuning, and model selection will be crucial in optimizing performance for this dataset.

## Decision Tree Interpretation

The decision tree model provides insight into different customer segments, guiding targeted marketing strategies:

#### Segment 1: Low Employment, Short Call Duration
- **Characteristics**: Customers in regions with low employment rates and short call durations.
- **Behavior**: Less likely to subscribe unless call duration is slightly longer.
- **Recommendations**: Engage customers by extending conversations and focusing on understanding their needs.

#### Segment 2: Low Employment, Long Call Duration
- **Characteristics**: Customers in low employment regions with longer call durations.
- **Behavior**: More likely to subscribe, especially if there have been few prior contacts.
- **Recommendations**: Use the extended call time to build a strong connection and present term deposits as a secure investment option.

#### Segment 3: High Employment, Short Call Duration
- **Characteristics**: Customers in high employment regions but with short call durations.
- **Behavior**: Less likely to subscribe, especially with low economic confidence.
- **Recommendations**: Address concerns early in the conversation and suggest follow-up emails or digital resources.

#### Segment 4: High Employment, Long Call Duration with Favorable Economic Conditions
- **Characteristics**: Customers in regions with high employment, long call durations, and favorable economic conditions.
- **Behavior**: More likely to subscribe, particularly if economic indicators are positive.
- **Recommendations**: Emphasize the benefits of term deposits and introduce related financial products.

#### Segment 5: High Employment, Long Call Duration with Unfavorable Economic Conditions

- **Characteristics**: Customers in regions with high employment, long call durations, but facing less favorable economic indicators.
- **Behavior**: Decision to subscribe is nuanced and influenced by economic conditions.
- **Recommendations**: Provide products with flexibility and stress the security of the investment.

## Marketing Strategies

  - **Tailored Campaigns:** Develop targeted marketing campaigns based on customer segments, focusing on their specific needs and preferences.
  - **Call Duration Optimization:** Implement strategies to increase call duration for segments with lower conversion rates.
  - **Economic Indicators:** Continuously monitor economic indicators and adjust marketing strategies accordingly.
  - **Agent Training:** Equip sales agents with knowledge about different customer segments and how to tailor their approach.
  - **Customer Relationship Management (CRM):** Utilize CRM systems to track customer interactions, preferences, and purchase history for personalized marketing.

## Deployment

### Model Deployment:

- The final model is ready for deployment, allowing the bank to predict the likelihood of clients subscribing to long-term deposits. The model can be integrated into the bank's CRM system for real-time decision-making.

## Usage

### Requirements:

- Python 3.x
- pandas, numpy, scikit-learn, matplotlib, seaborn (Python libraries)

### Running the Project:

**Clone the Repository:**
   ```bash
   git clone https://github.com/mitbans/Bank-Marketing-Campaigns.git
   ```

### Further Steps:
- Deploy the model using a web application framework like Flask or Streamlit for user-friendly interaction.

## Repository Structure
- <code>data/bank-additional-full.csv</code>: Contains dataset used in the analysis.
- <code>images/decision_tree.pdf</code>: Contains final Decision Tree Model.
- <code>notebooks/Predicting-Long-Term-Deposit-Success.ipynb</code>: Jupyter notebook with code for data analysis.
- <code>README.md</code>: Summary of findings and link to notebook

## Notebook
The detailed analysis and code can be found in the Jupyter notebook <a href="https://github.com/mitbans/Bank-Marketing-Campaigns/blob/main/notebooks/Predicting-Long-Term-Deposit-Success.ipynb">here</a>.

---

