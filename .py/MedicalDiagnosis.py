# Import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load CSV
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")
df = df.drop(["Outcome Variable"], axis = 1)
min_samples = 3
df = df[df["Disease"].map(df["Disease"].value_counts()) >= min_samples].reset_index(drop=True)

# Encoding Yes/No values
yes_no_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
for col in yes_no_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Encoding ordinal features (Low, Normal, High)
ordinal_map = {"Low": 0, "Normal": 1, "High": 2}
df["Blood Pressure"] = df["Blood Pressure"].map(ordinal_map)
df["Cholesterol Level"] = df["Cholesterol Level"].map(ordinal_map)

# Encoding gender
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])  # Male = 1, Female = 0

# Encoding diseases
disease_encoder = LabelEncoder()
df["Disease"] = disease_encoder.fit_transform(df["Disease"])

# Declaring labels
X = df.drop(["Disease"], axis = 1)
y = df["Disease"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.2)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Validation functions
def ask_yes_no(question):
    while True:
        answer = input(f"{question} (Yes/No): ").strip().title()
        if answer in ["Yes", "No"]:
            return answer
        print("Please enter Yes or No.")

def ask_choice(question, choices):
    choices = [c.title() for c in choices]
    while True:
        answer = input(f"{question} ({'/'.join(choices)}): ").strip().title()
        if answer in choices:
            return answer
        print(f"Please enter one of the following: {', '.join(choices)}.")

def ask_int(question, min_val=0, max_val=120):
    while True:
        try:
            value = int(input(f"{question}: ").strip())
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")

# Predict Diseases
def predict_disease_interactive(top_n = 3):
    fever = ask_yes_no("Do you have a fever?")
    cough = ask_yes_no("Do you have a cough?")
    fatigue = ask_yes_no("Are you feeling fatigued?")
    breathing = ask_yes_no("Do you have difficulty breathing?")
    gender = ask_choice("Gender?", ["male", "female"])
    age = ask_int("Age?")
    bp = ask_choice("Blood Pressure?", ["low", "normal", "high"])
    cholesterol = ask_choice("Cholesterol?", ["low", "normal", "high"])
    
    yes_no = {"Yes": 1, "No": 0}
    ordinal_map = {"Low": 0, "Normal": 1, "High": 2}
    
    input_data = {
        "Fever": yes_no.get(fever, 0),
        "Cough": yes_no.get(cough, 0),
        "Fatigue": yes_no.get(fatigue, 0),
        "Difficulty Breathing": yes_no.get(breathing, 0),
        "Age": age,
        "Gender": gender_encoder.transform([gender])[0],
        "Blood Pressure": ordinal_map.get(bp, 1),
        "Cholesterol Level": ordinal_map.get(cholesterol, 1)
    }
    
    input_df = pd.DataFrame([input_data])
    probs = model.predict_proba(input_df)[0]
    top_indices = np.argsort(probs)[-top_n:][::-1]
    
    print("\nTop predicted diseases:")
    for i in top_indices:
        disease = disease_encoder.inverse_transform([i])[0]
        prob = probs[i] * 100
        print(f" - {disease}: {prob:.2f}%")

# User prompt
while True:
    user_input = input("Enter anything to begin (0 to quit):")
    if user_input == "0":    
        break
    else:
        predict_disease_interactive()