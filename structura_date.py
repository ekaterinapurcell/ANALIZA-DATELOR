import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurăm stilul pentru grafice
sns.set(style="whitegrid")

# Încărcăm datele din fișierul CSV
df = pd.read_csv('/Users/ecaterinna/Desktop/UTM/an-3/AD/analizadatelor/maintenance.csv')

# Afișăm informații despre dataset: tipurile de date, numărul de observații
print("Informații despre dataset:")
df.info()  

# Afișăm statistici descriptive pentru variabilele numerice
print("\nStatistici descriptive pentru variabilele numerice:")
print(df.describe())

# ---------------------------------------------------------------------
# Tabel sumar pentru a vizualiza structura dataset-ului

# Dacă coloana este numerică, se vor adăuga și statistici descriptive suplimentare:
#   - Media, deviația standard, minim, percentila 25%, mediană, percentila 75%, maxim
# ---------------------------------------------------------------------

# Inițializăm o listă în care vom stoca informațiile pentru fiecare coloană
summary_list = []

# Iterăm prin fiecare coloană din DataFrame
for col in df.columns:
    col_data = {}
    col_data['Coloana'] = col
    col_data['Tip Date'] = df[col].dtype
    col_data['Valori Nenule'] = df[col].count()
    col_data['Valori Lipsă'] = df[col].isnull().sum()
    col_data['Valori Unice'] = df[col].nunique()
    
    # Dacă coloana este numerică, adăugăm statistici descriptive suplimentare
    if pd.api.types.is_numeric_dtype(df[col]):
        col_data['Medie']     = df[col].mean()
        col_data['Deviație']  = df[col].std()
        col_data['Minim']     = df[col].min()
        col_data['25%']       = df[col].quantile(0.25)
        col_data['Mediană']   = df[col].median()
        col_data['75%']       = df[col].quantile(0.75)
        col_data['Maxim']     = df[col].max()
    else:
        # Pentru coloanele non-numerice, completăm cu None valorile care nu se aplică
        col_data['Medie']     = None
        col_data['Deviație']  = None
        col_data['Minim']     = None
        col_data['25%']       = None
        col_data['Mediană']   = None
        col_data['75%']       = None
        col_data['Maxim']     = None

    summary_list.append(col_data)

# Convertim lista de dicționare într-un DataFrame pentru a avea o vizualizare tabulară
summary_df = pd.DataFrame(summary_list)

# Afișăm tabelul sumar
print("\nTabelul sumar al dataset-ului:")
print(summary_df)
