import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from example import new_patient
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
pd.set_option('display.max_columns', None)
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['stroke'].value_counts())
sns.countplot(x='stroke', data=df)
plt.title("stroke:1, not stroke:0")
plt.show()
df = df.drop(columns=['id'])
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(df.dtypes)
print("any objects left?", df.select_dtypes(include='object').shape[1])
X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)
print("NaNs in X_train:", np.isnan(X_train).sum())
print("NaNs in X_test:", np.isnan(X_test).sum())
print(df.isna().sum())
num_cols = ['age', 'avg_glucose_level', 'bmi']
preprocess = ColumnTransformer(transformers=[('scale', StandardScaler(), num_cols)], remainder='passthrough')
missing = [c for c in num_cols if c not in X.columns]
print("missing numeric columns:", missing)
loreg=LogisticRegression(max_iter=2000, class_weight='balanced')
pipe = Pipeline(steps=[('prep', preprocess), ('clf', loreg)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfuaion_matrix:", confusion_matrix(y_test, y_pred))
print("\nreport:", classification_report(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)
plt.figure(figsize=(7, 6))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, label=f'ROC curve, AUC={roc_auc:.3f})')
plt.title('ROC curve')
plt.legend()
plt.show()
#finding out which feature has more effrect
logreg_model = pipe.named_steps['clf']
coefs = logreg_model.coef_[0]
features = X_train.columns
importance = pd.DataFrame({'feature': features, 'coef': coefs, 'abs_coef': np.abs(coefs)})
print(importance)
print("coefs shape:", coefs.shape)
print("num features:", len(features))
#testing the code on a patient
pipe.predict(new_patient)
pipe.predict_proba(new_patient)
prob = pipe.predict_proba(new_patient)[0][1]
pred_label = pipe.predict(new_patient)[0]


print("\n---------------------------------")
print(" PATIENT STROKE RISK REPORT")
print("---------------------------------")

if pred_label == 1:
    print("Prediction: HIGH RISK (Stroke Likely)")
else:
    print("Prediction: LOW RISK (No Stroke)")

print(f"Probability of Stroke: {prob*100:.2f}%")
print("---------------------------------")
