# Vizualizarea asocierilor între variabilele numerice 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setăm stilul pentru grafice
sns.set(style="whitegrid")

# Creăm directorul 'scatter_plots' dacă nu există deja
output_dir = "scatter_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Încărcăm datele din fișierul CSV
df = pd.read_csv('/Users/ecaterinna/Desktop/UTM/an-3/AD/analizadatelor/maintenance.csv')

# Identificăm variabilele numerice relevante
num_vars = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

# Generăm scatter plot-uri pentru fiecare pereche unică de variabile numerice
for i in range(len(num_vars)):
    for j in range(i + 1, len(num_vars)):
        var1 = num_vars[i]
        var2 = num_vars[j]
        
        plt.figure(figsize=(8, 5))
        ax = sns.scatterplot(data=df, x=var1, y=var2, color='coral')
        ax.set_title(f'Asocierea între {var1} și {var2}', fontsize=14)
        ax.set_xlabel(var1, fontsize=12)
        ax.set_ylabel(var2, fontsize=12)
        
        # Adăugăm o descriere sub grafic
        plt.figtext(0.5, 0.01, 
                    f'Scatter plot care ilustrează relația dintre variabilele "{var1}" și "{var2}".',
                    wrap=True, horizontalalignment='center', fontsize=10)
        
        # Ajustăm layout-ul pentru a include descrierea
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # Salvăm figura în directorul "scatter_plots" cu un nume specific
        filename = os.path.join(output_dir, f"scatter_{var1.replace(' ', '_')}_vs_{var2.replace(' ', '_')}.png")
        plt.savefig(filename)
        
        # Închidem figura pentru a elibera memoria
        plt.close()

