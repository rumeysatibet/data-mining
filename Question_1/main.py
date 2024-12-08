import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Uyarıları susturma
warnings.filterwarnings("ignore")

# Veri setini yükleme
df = pd.read_csv("output.csv")

# Veri keşfi
print("Veri setinin ilk 5 satırı:")
print(df.head())
print("\nVeri seti sütunları:")
print(df.columns)

# Eksik verileri kontrol etme ve temizleme
print("\nEksik değerler:")
print(df.isnull().sum())
df = df.dropna()  # Eksik değer içeren satırları kaldırma

# Kategorik sütunları kodlama
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])  # "Sex" sütununu sayısallaştırma
df["Anaemic"] = le.fit_transform(df["Anaemic"])  # "Anaemic" hedef değişkenini sayısallaştırma

# Özellik ve hedef değişken ayrımı
target_column = "Anaemic"  # Hedef sütun
X = df.drop([target_column, "Number"], axis=1)  # Özellikler (Number sütunu analiz için gereksiz)
y = df[target_column]  # Hedef değişken

# Veri setini bölme
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Özellikleri standardize etme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Modeller
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='linear', random_state=42)
}

# Performans metriklerini hesaplama
results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict(X_val_scaled)

    # Doğrulama seti metrikleri
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    results[model_name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

    print(f"{model_name} - Validation Metrics:")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Test seti sonuçları ve görselleştirme
for model_name, model in models.items():
    y_test_pred = model.predict(X_test_scaled)

    # Test seti metrikleri
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print(f"{model_name} - Test Metrics:")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()

# Model performans karşılaştırması
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
for metric in metrics:
    values = [results[model][metric] for model in results]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=values, palette='viridis')
    plt.title(f'{metric} Comparison Across Models')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f'{metric.lower()}_comparison.png')
    plt.show()
