import pandas as pd

# Charger le dataset original
input_path = "data/raw/mlops_dataset_original.csv"
output_path = "data/raw/mlops_dataset_v3.csv"

# Lire toutes les lignes
df = pd.read_csv(input_path)

# Sauvegarder la nouvelle version
df.to_csv(output_path, index=False)
print(f"Fichier {output_path} généré avec toutes les lignes de l'original.")
