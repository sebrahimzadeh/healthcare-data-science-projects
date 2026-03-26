import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load COVID-19 dataset 
data = pd.read_csv(r"C:\Users\sebra\Desktop\covid_data.csv")  

# Example columns (adjust to your CSV)
columns = ["Age", "Gender", "Comorbidity_Score", "Oxygen_Level", "Hospitalized", "Outcome"]
data.columns = columns

# Features & target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
