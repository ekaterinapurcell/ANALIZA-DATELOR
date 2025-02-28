import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Setare backend pentru evitarea problemelor pe MacOS
import matplotlib
matplotlib.use('Agg')

# PAS 1: Încărcarea dataset-ului
df = pd.read_csv('/Users/ecaterinna/Desktop/UTM/an-3/AD/analizadatelor/maintenance.csv')

# PAS 2: Selectarea caracteristicilor și variabilei dependente
X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
Y = df['Machine failure']

# PAS 3: Selecția celor mai importante caracteristici
selector = SelectKBest(f_classif, k=4)  # Alegem cele mai relevante 4 caracteristici
X_selected = selector.fit_transform(X, Y)

# PAS 4: Împărțirea în seturi de antrenament și testare
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=42, stratify=Y)

# PAS 5: Standardizarea caracteristicilor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PAS 6: Optimizarea hiperparametrilor cu `RandomizedSearchCV`
param_dist = {
    'n_estimators': [50, 100, 150],  # Reducem numărul de arbori
    'max_depth': [5, 10, 15],  # Testăm doar câteva valori
    'min_samples_split': [2, 5],  # Reducem numărul de opțiuni
    'min_samples_leaf': [1, 2],  # Testăm doar 2 opțiuni
    'class_weight': ['balanced']
}

random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,  # **Mult mai rapid decât GridSearchCV!**
    cv=StratifiedKFold(n_splits=5),  # **Reducem la 5 fold-uri**
    scoring='f1',  # Optimizăm pentru echilibru între precizie și recall
    n_jobs=-1,
    random_state=42
)
random_search_rf.fit(X_train_scaled, Y_train)

# PAS 7: Antrenarea modelului îmbunătățit
rf_model_best = random_search_rf.best_estimator_
rf_model_best.fit(X_train_scaled, Y_train)

# PAS 8: Determinarea threshold-ului optim
Y_pred_proba = rf_model_best.predict_proba(X_test_scaled)[:, 1]
thresholds = np.linspace(0.1, 0.9, 20)  # Testăm mai multe praguri între 0.1 și 0.9

best_threshold = 0.5
best_f1 = 0
for t in thresholds:
    Y_pred_test = (Y_pred_proba >= t).astype(int)
    f1 = precision_score(Y_test, Y_pred_test, zero_division=0) * recall_score(Y_test, Y_pred_test, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# PAS 9: Predicții și evaluare cu threshold-ul optim
Y_pred = (Y_pred_proba >= best_threshold).astype(int)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, zero_division=0)
recall = recall_score(Y_test, Y_pred, zero_division=0)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)

# PAS 10: Crearea dashboard-ului
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Matricea de confuzie
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Nu defectat', 'Defectat'], yticklabels=['Nu defectat', 'Defectat'], ax=axs[0, 0])
axs[0, 0].set_title(f'Matricea de Confuzie - Random Forest (Threshold={best_threshold:.2f})')
axs[0, 0].set_xlabel('Predicții')
axs[0, 0].set_ylabel('Realitate')

# Precizie și Recall
axs[1, 0].bar(['Precizie', 'Recall'], [precision, recall], color=['blue', 'orange'])
axs[1, 0].set_title('Precizie și Recall - Random Forest')
axs[1, 0].set_ylabel('Scor')

# Raportul de clasificare
axs[1, 1].axis('off')
axs[1, 1].text(0.1, 0.5, classification_rep, fontsize=12)

# PAS 11: Curba ROC
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
roc_auc = auc(fpr, tpr)

axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
axs[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[0, 1].set_title('Curba ROC - Random Forest')
axs[0, 1].set_xlabel('False Positive Rate')
axs[0, 1].set_ylabel('True Positive Rate')
axs[0, 1].legend(loc='lower right')

# Salvarea dashboard-ului
plt.tight_layout()
plt.savefig('dashboard_M_RForest.png')

print(f"Dashboard-ul a fost salvat ca 'shboard_M_RForest.png'.")
