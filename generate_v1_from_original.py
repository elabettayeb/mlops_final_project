import pandas as pd

# Charger le dataset original
input_path = "data/raw/mlops_dataset_original.csv"
output_path = "data/raw/mlops_dataset_v1_augmented.csv"

# Lire les 401 premières lignes
n_rows = 401
df = pd.read_csv(input_path, nrows=n_rows)

# Sauvegarder la nouvelle version
df.to_csv(output_path, index=False)
print(f"Fichier {output_path} généré avec les {n_rows} premières lignes.")
