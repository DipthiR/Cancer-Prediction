import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# -------------------------------
# 1. Load and Preprocess Data
# -------------------------------

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Encode categorical columns
    label_encoders = {}
    categorical_cols = ['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiat']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Fill missing values (if any)
    df = df.fillna(df.mean(numeric_only=True))

    return df, label_encoders

# -------------------------------
# 2. Train Models
# -------------------------------

def train_models(df):
    # Define features for both models
    features = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

    # Stage Classification (Assume 'Class' is the target for cancer stage)
    X_class = df[features]
    y_class = df['Class']  # Target: Cancer Stage
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    class_model = RandomForestClassifier()
    class_model.fit(X_train_class, y_train_class)

    # Store models and test data
    models = {
        'class': (class_model, X_test_class, y_test_class),
    }

    return models

# -------------------------------
# 3. Evaluate Models
# -------------------------------

def evaluate_classification_model(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    print(f"\n--- {title} Classification Report ---")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {title}")
    plt.show()

# -------------------------------
# 4. Predict for New Patient
# -------------------------------

def predict_new_patient(models, input_data):
    input_array = np.array(input_data).reshape(1, -1)

    # Load model for cancer stage prediction
    class_model = models['class'][0]

    # Predict cancer stage
    predicted_class = class_model.predict(input_array)[0]

    # Define treatment based on cancer stage (manually set rules)
    treatment_map = {
        1: "Surgery",
        2: "Surgery + Radiation",
        3: "Chemotherapy + Surgery",
        4: "Chemotherapy + Radiation + Surgery"
    }

    predicted_treatment = treatment_map.get(predicted_class, "Unknown Treatment")

    # Output the predictions
    print("\n--- Patient Prediction ---")
    print(f"🔍 Cancer Stage: Stage {predicted_class}")
    print(f"💊 Recommended Treatment: {predicted_treatment}")

# -------------------------------
# 5. Main Execution
# -------------------------------

def main():
    # Load data and preprocess
    df, label_encoders = load_and_preprocess_data("C:\\Users\\dipthi\\Downloads\\cancer.csv")
    
    # Train the models
    models = train_models(df)

    # Evaluate the models
    evaluate_classification_model(*models['class'], title="Cancer Stage")

    # Example Patient Input (Replace with dynamic input if needed)
    new_patient_data = [60, 1, 4, 5, 1, 3, 3, 2, 0]  # Example values (age, menopause, tumor-size, inv-nodes, etc.)
    predict_new_patient(models, new_patient_data)

if __name__ == '__main__':
    main()
