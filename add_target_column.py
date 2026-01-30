import pandas as pd
import numpy as np

# Charger le dataset
input_path = "data/raw/mlops_dataset_v3.csv"
output_path = "data/raw/mlops_dataset_v3.csv"

# Charger le CSV

df = pd.read_csv(input_path)

# Générer une colonne cible fictive (exemple : basée sur num_users, desc_length, is_touristique)
np.random.seed(42)
df["visiteurs"] = (
    df["num_users"] * 100
    + df["desc_length"] * 10
    + df["is_touristique"] * 500
    + np.random.randint(0, 1000, size=len(df))
)

# Arrondir et convertir en int

df["visiteurs"] = df["visiteurs"].astype(int)

# Sauvegarder le dataset mis à jour (remplace l'ancien)
df.to_csv(output_path, index=False)
print(f"Colonne 'visiteurs' ajoutée et dataset mis à jour dans {output_path}")
