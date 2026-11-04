from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "Model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"

print("START")

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data folder not found at {DATA_DIR}")

csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

known_names = {"heart.csv", "heart.csv.csv"}
DATA_PATH = next((p for p in csv_files if p.name in known_names), csv_files[0])

# Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
print(f"DATA LOADED from {DATA_PATH}")
# Features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("MODEL TRAINED")

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"MODEL SAVED SUCCESSFULLY to {MODEL_PATH}")