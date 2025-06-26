import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Create dummy data
np.random.seed(42)
data = []
labels = []

for _ in range(100):
    age = np.random.randint(18, 45)
    employment = np.random.choice([0, 1])
    children = np.random.choice([0, 1, 2, 3, 4])
    pregnancy = np.random.choice([0, 1])
    delivery = np.random.choice([0, 1])
    support = np.random.choice([0, 1, 2])
    epds = np.random.randint(0, 4, size=10)

    features = [age, employment, children, pregnancy, delivery, support] + epds.tolist()
    score = sum(epds)

    # Label based on total EPDS score
    if score < 10:
        label = "Mild"
    elif score < 14:
        label = "Moderate"
    elif score < 18:
        label = "Severe"
    else:
        label = "Profound"

    data.append(features)
    labels.append(label)

# Prepare data
columns = ["Age", "Employment", "Children", "Pregnancy", "Delivery", "FamilySupport"] + [f"Q{i}" for i in range(1, 11)]
df = pd.DataFrame(data, columns=columns)

# Encode target
le = LabelEncoder()
y = le.fit_transform(labels)

# Train model
model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', max_iter=500, random_state=42)
model.fit(df, y)

# Save files
joblib.dump(model, "ffnn_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model and label encoder saved.")
