import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Încărcarea datelor
df = pd.read_csv('/Users/ecaterinna/Desktop/UTM/an-3/AD/analizadatelor/maintenance.csv')

# Crearea directorului pentru salvarea boxploturilor
output_dir = "boxplots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Definirea variabilelor pentru care dorim să generăm boxploturi
variables = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target_variable = 'Machine failure'

# Crearea boxploturilor pentru fiecare variabilă explicativă în raport cu 'Machine failure'
for var in variables:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=target_variable, y=var, data=df, palette="Blues")
    
    plt.title(f'Compararea {var} în funcție de Machine failure', fontsize=14)
    plt.xlabel('Machine failure', fontsize=12)
    plt.ylabel(var, fontsize=12)
    
    # Salvarea boxplotului
    filename = os.path.join(output_dir, f"boxplot_{var.replace(' ', '_')}_vs_{target_variable}.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
