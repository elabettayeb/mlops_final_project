import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Charger le dataset
input_path = "data/raw/mlops_dataset_v3.csv"
df = pd.read_csv(input_path)

# Features à utiliser (toutes sauf la cible et les non-numériques inutiles)
features = [
    col for col in df.columns
    if col not in ["city_name", "country", "description", "region", "visiteurs"]
]

X = df[features]
y = df["visiteurs"]

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde du modèle
joblib.dump(model, "model.pkl")
print("Modèle entraîné et sauvegardé dans model.pkl")
