import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")
df.dropna(inplace=True)

# Features and target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Encode the disease names
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', max_depth=10)
model.fit(X_train, y_train)

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "symptom_list.pkl")

print("✅ Model trained and saved successfully.")
print("✅ Accuracy:", model.score(X_test, y_test))
