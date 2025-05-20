import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
train_X = train.iloc[:, :-1]  # First 132 columns as features
train_y = train.iloc[:, -1]   # Last column as target (disease names)
test_X = test.iloc[:, :-1]
test_y = test.iloc[:, -1]
label_encoder = LabelEncoder()
y_encoded_train = label_encoder.fit_transform(train_y)
y_encoded_test = label_encoder.transform(test_y)
model = RandomForestClassifier(n_estimators=1, random_state=0)
model.fit(train_X, y_encoded_train)
pred_y = model.predict(test_X)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_encoded_test, pred_y, average="weighted"):
    print(f"Accuracy: {accuracy_score(y_encoded_test, pred_y):.1f}")
    print(f"Precision: {precision_score(y_encoded_test, pred_y, average=average):.1f}")
    print(f"Recall: {recall_score(y_encoded_test, pred_y, average=average):.1f}")
    print(f"F1-Score: {f1_score(y_encoded_test, pred_y, average=average):.1f}")

evaluate_model(y_encoded_test, pred_y)
joblib.dump(model, "trained_model")
feature_importance = model.feature_importances_
features = np.array(train_X.columns)

sorted_idx = np.argsort(feature_importance)[::-1]
top_n = 10
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx][:top_n], y=features[sorted_idx][:top_n])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top Feature Importance (Random Forest)")
plt.show()

#dly

import gradio as gr
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load trained model and label encoder
model = joblib.load("trained_model")
train = pd.read_csv("Training.csv")
feature_names = train.columns[:-1]  # First 132 columns
label_encoder = LabelEncoder()
label_encoder.fit(train.iloc[:, -1])  # Fit encoder on disease names

# Define prediction function
def predict_disease(*input_features):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    return f"Predicted Disease: {predicted_label}"

# Dynamically create inputs for Gradio interface
inputs = [gr.Slider(0, 1, step=1, label=feature) for feature in feature_names]

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_disease,
    inputs=inputs,
    outputs="text",
    title="Disease Prediction from Symptoms",
    description="Input 0 or 1 for each symptom. Model predicts the most likely disease."
)

interface.launch()