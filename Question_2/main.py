import os
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Loky uyarısını çözmek için ortam değişkeni ve uyarı filtresi
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Fiziksel çekirdek sayınızı girin
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# 1. Veri Ön İşleme
def preprocess_data(raw_data):
    # Gereksiz sütunları kaldırma
    cleaned_data = raw_data.drop(columns=["Timestamp", "Course", "YearOfStudy"])

    # Cinsiyet kodlama
    label_encoder = LabelEncoder()
    cleaned_data["Gender"] = label_encoder.fit_transform(cleaned_data["Gender"])  # Female = 0, Male = 1

    # Eksik veri kontrolü
    if cleaned_data.isnull().sum().sum() > 0:
        cleaned_data = cleaned_data.dropna()  # Eksik satırları kaldır

    # Sayısal sütunların normalizasyonu
    numerical_cols = ["Age", "CGPA", "SymptomFrequency_Last7Days", "SleepQuality",
                      "StudyStressLevel", "StudyHoursPerWeek", "AcademicEngagement"]
    scaler = StandardScaler()
    cleaned_data[numerical_cols] = scaler.fit_transform(cleaned_data[numerical_cols])

    return cleaned_data


# 2. Optimal Küme Sayısını Belirleme
def find_optimal_k(features_data, k_range):
    silhouette_scores = []
    inertia = []

    for k in k_range:
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans_model.fit_predict(features_data)
        silhouette_scores.append(silhouette_score(features_data, cluster_labels))
        inertia.append(kmeans_model.inertia_)

    # Silhouette Skoru Grafiği
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title('Silhouette Skoru ile Küme Sayısı (k)')
    plt.xlabel('Küme Sayısı (k)')
    plt.ylabel('Silhouette Skoru')
    plt.grid()
    plt.savefig('silhouette_score_plot.png')
    plt.show()

    # Elbow Yöntemi Grafiği
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Yöntemi ile Küme Sayısı (k)')
    plt.xlabel('Küme Sayısı (k)')
    plt.ylabel('Inertia')
    plt.grid()
    plt.savefig('elbow_method_plot.png')
    plt.show()

    optimal_k_value = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal Küme Sayısı (Silhouette): {optimal_k_value}")
    return optimal_k_value


# 3. K-means Kümeleme
def perform_kmeans(features_data, num_clusters):
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans_model.fit_predict(features_data)
    return kmeans_model, cluster_labels


# 4. Sonuçların Görselleştirilmesi
def visualize_clusters(features_data, cluster_labels_data):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_data)

    plt.figure(figsize=(8, 5))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels_data, cmap='viridis')
    plt.title('K-means Kümeleri (PCA ile 2D)')
    plt.xlabel('Bileşen 1')
    plt.ylabel('Bileşen 2')
    plt.colorbar(label='Cluster')
    plt.savefig('pca_clusters.png')
    plt.show()


# 5. Küme Merkezleri ve Analiz
def analyze_clusters(kmeans_model, features_data, cleaned_data, cluster_labels_data):
    # Küme Merkezleri
    cluster_centers = pd.DataFrame(kmeans_model.cluster_centers_, columns=features_data.columns)
    print("Küme Merkezleri:\n", cluster_centers)

    # Kümelerin Özellik Ortalamaları
    cleaned_data['Cluster'] = cluster_labels_data
    cluster_summary = cleaned_data.groupby('Cluster').mean()
    print("Kümelerin Özellik Ortalamaları:\n", cluster_summary)

    return cluster_centers, cluster_summary


# 6. Değerlendirme
def evaluate_clustering(features_data, cluster_labels_data):
    silhouette_avg = silhouette_score(features_data, cluster_labels_data)
    db_score = davies_bouldin_score(features_data, cluster_labels_data)
    print(f"Silhouette Skoru: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {db_score}")
    return silhouette_avg, db_score


# Ana Akış
if __name__ == "__main__":
    # Veri Yükleme
    data_path = 'mentalhealth_dataset.csv'
    raw_data = pd.read_csv(data_path)

    # Veri Ön İşleme
    preprocessed_data = preprocess_data(raw_data)

    # Özellik Seçimi
    selected_features = preprocessed_data.drop(columns=["Depression", "Anxiety", "PanicAttack",
                                                        "SpecialistTreatment", "HasMentalHealthSupport"])

    # Optimal K Seçimi
    k_range_values = range(2, 11)
    optimal_k_value = find_optimal_k(selected_features, k_range_values)

    # K-means Kümeleme
    kmeans_model, cluster_labels_result = perform_kmeans(selected_features, optimal_k_value)

    # Sonuçların Görselleştirilmesi
    visualize_clusters(selected_features, cluster_labels_result)

    # Küme Analizi
    cluster_centers_result, cluster_summary_result = analyze_clusters(kmeans_model, selected_features,
                                                                      preprocessed_data, cluster_labels_result)

    # Değerlendirme
    silhouette_avg_result, db_score_result = evaluate_clustering(selected_features, cluster_labels_result)

    print("Tüm işlemler başarıyla tamamlandı!")
