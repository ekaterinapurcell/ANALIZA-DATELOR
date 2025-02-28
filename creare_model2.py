import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE

# Setare backend pentru evitarea problemelor pe MacOS
import matplotlib
matplotlib.use('Agg')

# PAS 1: Încărcarea dataset-ului
df = pd.read_csv('/Users/ecaterinna/Desktop/UTM/an-3/AD/analizadatelor/maintenance.csv')

# PAS 2: Selectarea caracteristicilor și variabilei dependente
X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
Y = df['Machine failure']

# PAS 3: Feature Engineering (Adăugăm interacțiuni & termeni pătrați)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# PAS 4: Împărțirea în seturi de antrenament și testare
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42, stratify=Y)

# PAS 5: Standardizarea caracteristicilor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PAS 6: Feature Selection cu Recursive Feature Elimination (RFE)
logistic_base = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(logistic_base, n_features_to_select=5)  # Alegem cele mai relevante 5 variabile
X_train_selected = rfe.fit_transform(X_train_scaled, Y_train)
X_test_selected = rfe.transform(X_test_scaled)

# PAS 7: Optimizarea hiperparametrilor cu GridSearchCV
param_grid = {
    'C': np.logspace(-4, 4, 30),  # Testăm mai multe valori pentru regularizare
    'penalty': ['l1', 'l2'],  # Testăm atât L1, cât și L2
    'solver': ['liblinear'],  # Suportă ambele tipuri de regularizare
    'class_weight': ['balanced']  # Echilibrarea claselor
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=10),
    scoring='precision',  # 🌟 Optimizăm pentru precizie!
    n_jobs=-1
)
grid_search.fit(X_train_selected, Y_train)

# PAS 8: Antrenarea modelului îmbunătățit
logistic_model_best = grid_search.best_estimator_
logistic_model_best.fit(X_train_selected, Y_train)

# PAS 9: Ajustarea threshold-ului
threshold = 0.4  # 🌟 Setăm un prag mai mare pentru a reduce false positive
Y_pred_proba = logistic_model_best.predict_proba(X_test_selected)[:, 1]
Y_pred = (Y_pred_proba >= threshold).astype(int)

# PAS 10: Predicții și evaluare
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, zero_division=0)
recall = recall_score(Y_test, Y_pred, zero_division=0)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)

# PAS 11: Crearea dashboard-ului
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Matricea de confuzie
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Nu defectat', 'Defectat'], yticklabels=['Nu defectat', 'Defectat'], ax=axs[0, 0])
axs[0, 0].set_title(f'Matricea de Confuzie - Regresie Logistică (Optimizare Precizie) (Threshold={threshold})')
axs[0, 0].set_xlabel('Predicții')
axs[0, 0].set_ylabel('Realitate')

# Corectarea graficului Precizie și Recall
if precision + recall > 0:  # Evităm eroarea dacă sunt zero
    axs[1, 0].bar(['Precizie', 'Recall'], [precision, recall], color=['blue', 'orange'])
axs[1, 0].set_title('Precizie și Recall - Regresie Logistică (Optimizare Precizie)')
axs[1, 0].set_ylabel('Scor')

# Raportul de clasificare
axs[1, 1].axis('off')
axs[1, 1].text(0.1, 0.5, classification_rep, fontsize=12)

# PAS 12: Curba ROC
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
roc_auc = auc(fpr, tpr)

axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
axs[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[0, 1].set_title('Curba ROC - Regresie Logistică (Optimizare Precizie)')
axs[0, 1].set_xlabel('False Positive Rate')
axs[0, 1].set_ylabel('True Positive Rate')
axs[0, 1].legend(loc='lower right')

# Salvarea dashboard-ului
plt.tight_layout()
plt.savefig('dashboard_M_REG_LOG2.png')

print(f"Dashboard-ul a fost salvat ca 'dashboard_M_REG_LOG2.png'.")
