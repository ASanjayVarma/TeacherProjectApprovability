
# Machine Learning Project: Decision Tree Classifier

This project involves building a decision tree classifier to predict the approval status of projects based on various features. The dataset includes features such as the number of previously posted projects by a teacher, total quantity, total price, and project grade categories.

## Project Overview

The project performs the following steps:
1. Data Preprocessing:
   - Encoding categorical variables using OneHotEncoder.
   - Standardizing numerical features.
2. Model Training:
   - Splitting the data into training and testing sets.
   - Training a decision tree classifier.
3. Model Evaluation:
   - Evaluating the model using accuracy score.
4. Predictions:
   - Making predictions on new test data.

## Dataset

The datasets used in this project are provided in the following Google Drive folder:
- [Training and Test Data](https://drive.google.com/drive/folders/1xpYkIXglCbZfnRZT2QVUKkvfwDSWJWG8)

## Requirements

The project requires the following Python libraries:
- pandas
- scikit-learn

You can install the required libraries using:
```bash
pip install pandas scikit-learn
```

## Code

The main steps of the project are detailed in the `TPAP.ipynb` Jupyter notebook. Below is a brief overview of the code:

### Data Preprocessing

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load training data
dataframe = pd.read_csv("train.csv")

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
columns_encoded = ['Grades PreK-2', 'Grades 3-5', 'Grades 6-8', 'Grades 9-12']
one_hot_encoded = encoder.fit_transform(dataframe[["project_grade_category"]])
dataframe_encoded = pd.DataFrame(one_hot_encoded, columns=columns_encoded)
new_dataframe = dataframe.join(dataframe_encoded)

# Select features and target
X_normal = new_dataframe[['teacher_number_of_previously_posted_projects', 'total_quantity', 'total_price', 'Grades PreK-2', 'Grades 3-5', 'Grades 6-8', 'Grades 9-12']]
y = new_dataframe[['project_is_approved']]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_normal)
X = pd.DataFrame(X)
```

### Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn import tree

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Train decision tree classifier
model = tree.DecisionTreeClassifier(random_state=9)
model.fit(X_train, y_train)
```

### Model Evaluation

```python
from sklearn.metrics import accuracy_score

# Predict on test data
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model Accuracy: {accuracy}")
```

### Predictions on New Data

```python
# Load test data
test_data = pd.read_csv('test.csv')

# Encode and standardize test data
one_hot_encoded = encoder.fit_transform(test_data[["project_grade_category"]])
test_data_encoded = pd.DataFrame(one_hot_encoded, columns=columns_encoded)
new_test_data = test_data.join(test_data_encoded)
X_new = scaler.fit_transform(new_test_data[["teacher_number_of_previously_posted_projects", "total_quantity", "total_price", "Grades PreK-2", "Grades 3-5", "Grades 6-8", "Grades 9-12"]])
X_new = pd.DataFrame(X_new)

# Make predictions
y_new_pred_test = model.predict(X_new)
print(y_new_pred_test)
```

## Usage

1. Clone the repository:
```bash
git clone <your-repository-url>
```

2. Navigate to the project directory and open the Jupyter notebook:
```bash
cd <project-directory>
jupyter notebook TPAP.ipynb
```

3. Follow the steps in the notebook to preprocess the data, train the model, evaluate its performance, and make predictions.

## License

This project is licensed under the MIT License.
