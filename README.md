# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the student placement dataset and preprocess it.
2. Split the data into training and testing sets.
3. Train the Logistic Regression model using training data.
4. Predict placement status and evaluate the model.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sugavelan S
RegisterNumber:  25005466
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")

df = df.drop(["sl_no", "salary"], axis=1)

df = df.dropna()
df = df.drop_duplicates()


le = LabelEncoder()
categorical_columns = [
    "gender", "ssc_b", "hsc_b", "hsc_s",
    "degree_t", "workex", "specialisation", "status"
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


report = classification_report(y_test, y_pred)

print("\n" + "="*60)
print("        LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("="*60)
print(report)
print("="*60)
print(f"Accuracy of the Model : {accuracy:.4f}")
print("="*60)


sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()

sample_input = pd.DataFrame(
    [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]],
    columns=X.columns
)

sample_prediction = model.predict(sample_input)
print("\nSample Prediction:", sample_prediction)
```

## Output:
<img width="1014" height="865" alt="image" src="https://github.com/user-attachments/assets/429ea396-976c-41ec-9ecd-5797cbedbc40" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
