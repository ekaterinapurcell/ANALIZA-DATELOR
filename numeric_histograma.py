import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setăm stilul pentru grafice
sns.set(style="whitegrid")

# Creăm directorul 'histograme'
output_dir = "histograme"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Încărcăm datele din fișierul CSV
df = pd.read_csv('/Users/ecaterinna/Desktop/UTM/an-3/AD/analizadatelor/maintenance.csv')

# Selectăm variabilele numerice
num_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Pentru fiecare variabilă numerică, generăm o histogramă și o salvăm ca fișier separat
for var in num_vars:
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(df[var], kde=False, color='skyblue', edgecolor='black')  # Setează kde=False
    plt.title(f'Distribuția numerică: {var}', fontsize=14)
    plt.xlabel(var, fontsize=12)
    plt.ylabel('Frecvența', fontsize=12)
    
    # Adăugăm o descriere sub grafic folosind plt.figtext
    plt.figtext(0.5, 0.01, f'Acest plot arată distribuția valorilor pentru variabila "{var}".', 
                wrap=True, horizontalalignment='center', fontsize=10)
    
    # Ajustăm layout-ul pentru a nu suprapune descrierea
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Salvăm figura în directorul "histograme" cu un nume specific
    filename = os.path.join(output_dir, f"histograma_{var}.png")
    plt.savefig(filename)
    print(f"Salvat: {filename}")
    plt.close()

# **Distribuția generală a defecțiunilor (Failure vs No Failure)**
failure_counts = df["Machine failure"].value_counts()

# **Distribuția pe tipuri de defecțiuni (HDF, OSF, etc.)**
failure_types = df.iloc[:, 9:].sum()

# Grafic pentru distribuția generală a defecțiunilor
plt.figure(figsize=(6, 4))
plt.bar(["Ne defectat", "Defectat"], failure_counts, color=['skyblue', 'skyblue'])
plt.xlabel("Status Defecțiune")
plt.ylabel("Număr Cazuri")
plt.title("Distribuția generală a defecțiunilor")
plt.tight_layout()
filename = os.path.join(output_dir, "distributia_defectiunilor.png")
plt.savefig(filename)
plt.show()

# Grafic pentru distribuția tipurilor de defecțiuni
plt.figure(figsize=(8, 4))
failure_types.plot(kind='bar', color='skyblue')
plt.xlabel("Tip de defecțiune")
plt.ylabel("Număr de cazuri")
plt.title("Distribuția tipurilor de defecțiuni")
plt.tight_layout()
filename = os.path.join(output_dir, "distributia_tipurilor_defectiunilor.png")
plt.savefig(filename)
plt.show()
