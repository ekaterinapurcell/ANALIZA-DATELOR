import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler

# Încărcați setul de date
df = pd.read_csv('/Users/ecaterinna/Desktop/UTM/an-3/AD/analizadatelor/maintenance.csv')

# Verifică primele 5 rânduri pentru a înțelege structura datelor
print(df.head())

# Verifică datele lipsă
print("\nDate lipsă în fiecare coloană:")
print(df.isnull().sum())

# Completarea valorilor lipsă
df['Air temperature [K]'] = df['Air temperature [K]'].fillna(df['Air temperature [K]'].mean())
df['Tool wear [min]'] = df['Tool wear [min]'].fillna(df['Tool wear [min]'].median())

# Împărțirea datelor în seturi de antrenament și testare
X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
Y = df['Machine failure']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Verifică dimensiunile fiecărui set de date
print(f"Dimensiunea setului de antrenament (X_train, Y_train): {X_train.shape}, {Y_train.shape}")
print(f"Dimensiunea setului de testare (X_test, Y_test): {X_test.shape}, {Y_test.shape}")

# Verifică distribuția valorilor "Machine failure" în seturile de antrenament și testare
print(f"Distribuția 'Machine failure' în setul de antrenament:\n{Y_train.value_counts()}")
print(f"Distribuția 'Machine failure' în setul de testare:\n{Y_test.value_counts()}")

# Vizualizarea distribuției "Machine failure" în setul de antrenament
plt.figure(figsize=(6, 4))
sns.countplot(x=Y_train)
plt.title('Distribuția Machine failure în setul de antrenament')
plt.xlabel('Machine failure')
plt.ylabel('Număr de observații')
plt.show()

# Scaled data: standardizarea caracteristicilor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crearea modelului de regresie logistică
model = LogisticRegression(max_iter=1000)  
model.fit(X_train_scaled, Y_train)

# Afișează coeficientul modelului pentru a verifica că s-a antrenat
coef = model.coef_[0]
features = X.columns
coef_df = pd.DataFrame(coef, features, columns=['Coeficient'])
print(coef_df)

# Predicțiile pe setul de testare
Y_pred = model.predict(X_test_scaled)

# Afișarea performanței modelului
print(f"Accuratețea modelului: {accuracy_score(Y_test, Y_pred)}")
print("Matricea de confuzie:\n", confusion_matrix(Y_test, Y_pred))
print("Raport de clasificare:\n", classification_report(Y_test, Y_pred))

# 1. Matricea de confuzie
cm = confusion_matrix(Y_test, Y_pred)

# 2. Curba de învățare
train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=5)

# 3. Precizia și Recall-ul
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)

# Crearea figurii pentru dashboard
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Matricea de confuzie
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Nu defectat', 'Defectat'], yticklabels=['Nu defectat', 'Defectat'], ax=axs[0, 0])
axs[0, 0].set_title('Matricea de Confuzie')
axs[0, 0].set_xlabel('Predicții')
axs[0, 0].set_ylabel('Realitate')

# Curba de învățare
axs[0, 1].plot(train_sizes, train_scores.mean(axis=1), label='Antrenament')
axs[0, 1].plot(train_sizes, test_scores.mean(axis=1), label='Test')
axs[0, 1].set_title('Curba de Învațare')
axs[0, 1].set_xlabel('Dimensiune Set de Antrenament')
axs[0, 1].set_ylabel('Performanță')
axs[0, 1].legend(loc='best')

# Precizia și Recall-ul
axs[1, 0].bar(['Precizie', 'Recall'], [precision, recall], color=['blue', 'orange'])
axs[1, 0].set_title('Precizie și Recall')
axs[1, 0].set_ylabel('Scor')

# Raportul de clasificare (ca text)
axs[1, 1].axis('off')  # Ascunde axele pentru a scrie textul
classification_rep = classification_report(Y_test, Y_pred)
axs[1, 1].text(0.1, 0.5, classification_rep, fontsize=12)

# Salvarea dashboard-ului ca imagine
plt.tight_layout()
plt.savefig('dashboard_M_REG_LOG1.png')
plt.show()

# Evaluarea curbei ROC și AUC
from sklearn.metrics import roc_curve, auc

# Calcularea curbei ROC
fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

# Vizualizarea curbei ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Salvarea curbei ROC ca imagine
roc_filename = 'roc_curve.png'
plt.savefig(roc_filename)
plt.show()

print(f"Curba ROC a fost salvată cu succes ca {roc_filename}")
