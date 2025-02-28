# Vizualizarea distribuției variabilelor categorice
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setăm stilul pentru grafice
sns.set(style="whitegrid")

# Creăm directorul 
output_dir = "categorice"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Încărcăm datele din fișierul CSV
df = pd.read_csv('/Users/ecaterinna/Desktop/UTM/an-3/AD/analizadatelor/maintenance.csv')

# Identificăm variabilele categorice relevante
cat_vars = ["Type"]

# Generăm bar plot-uri pentru variabilele categorice și le salvăm ca fișiere separate
for var in cat_vars:
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x=var, palette='viridis')
    ax.set_title(f'Distribuția categoriilor: {var}', fontsize=14)
    ax.set_xlabel(var, fontsize=12)
    ax.set_ylabel('Număr de observații', fontsize=12)
    plt.xticks(rotation=45)
    
    # Adăugăm o descriere sub grafic
    plt.figtext(0.5, 0.01, 
                f'Acest plot prezintă distribuția categoriilor pentru variabila "{var}".', 
                wrap=True, horizontalalignment='center', fontsize=10)
    
    # Ajustăm layout-ul pentru a include descrierea
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Salvăm figura în directorul "categorice" cu un nume specific
    filename = os.path.join(output_dir, f"barplot_{var}.png")
    plt.savefig(filename)
    
    # Închidem figura pentru a elibera memoria
    plt.close()

