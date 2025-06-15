import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load the dataset
df = pd.read_csv("dataset.csv")

# Step 2: Replace NaNs with None
df.fillna('None', inplace=True)

# Step 3: Extract all unique symptoms
symptom_columns = df.columns[:-1]  # all except Disease
all_symptoms = set()

for col in symptom_columns:
    all_symptoms.update(df[col].unique())

all_symptoms = sorted([s for s in all_symptoms if s != 'None'])

# Step 4: Create binary symptom matrix
X = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for index, row in df.iterrows():
    for symptom in row[symptom_columns]:
        if symptom != 'None':
            X.at[index, symptom] = 1

# Step 5: Encode Disease column
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

from sklearn.utils import resample

# Combine features and labels
df_combined = pd.concat([X, pd.Series(y, name="label")], axis=1)

# Separate majority and minority classes
max_class = df_combined['label'].value_counts().idxmax()
dfs = []

for label in df_combined['label'].unique():
    df_label = df_combined[df_combined['label'] == label]
    dfs.append(resample(df_label, 
                        replace=True, 
                        n_samples=len(df_combined[df_combined['label'] == max_class]), 
                        random_state=42))

df_balanced = pd.concat(dfs)
X = df_balanced.drop("label", axis=1)
y = df_balanced["label"]


# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 8: Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(list(X.columns), "symptom_list.pkl")

print("✅ Model trained and saved!")
print("🔍 Accuracy on test set:", model.score(X_test, y_test))
