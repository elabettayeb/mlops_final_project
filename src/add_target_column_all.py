import pandas as pd
import numpy as np

# Liste des fichiers à traiter
files = [
    "data/raw/mlops_dataset_v1_augmented.csv",
    "data/raw/mlops_dataset_v2_varied.csv",
    "data/raw/mlops_dataset_v3.csv"
]

for input_path in files:
    df = pd.read_csv(input_path)
    # Générer une colonne cible fictive (exemple : basée sur num_users, desc_length, is_touristique)
    np.random.seed(42)
    df["visiteurs"] = (
        df["num_users"] * 100
        + df["desc_length"] * 10
        + df["is_touristique"] * 500
        + np.random.randint(0, 1000, size=len(df))
    )
    df["visiteurs"] = df["visiteurs"].astype(int)
    df.to_csv(input_path, index=False)
    print(f"Colonne 'visiteurs' ajoutée et dataset mis à jour dans {input_path}")
