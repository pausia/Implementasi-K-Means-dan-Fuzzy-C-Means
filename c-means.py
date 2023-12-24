import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from IPython.display import display

# Load data
def load_data(file_name):
    full_path = r'C:\Users\ACER\Documents\A Semester 7\Logika Fuzzy\Pak Adha\Tugas-2\code\csv' + '\\' + file_name
    data = pd.read_csv(full_path)
    return data

# Fuzzy C-Means clustering
def fuzzy_cmeans_clustering(data, num_clusters, m=2, error=0.005, maxiter=1000):
    features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    # Standarisasi data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    # Reduksi dimensi menggunakan PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    
    # Inisialisasi centroid
    centroids = np.random.rand(num_clusters, reduced_data.shape[1])
    
    for i in range(maxiter):
        # Hitung jarak
        distances = np.linalg.norm(reduced_data[:, np.newaxis] - centroids, axis=2)
        
        # Hitung membership
        u = 1 / (distances / distances.sum(axis=1)[:, np.newaxis])**m
        u = u / u.sum(axis=1)[:, np.newaxis]
        
        # Hitung centroid baru
        centroids_new = u.transpose(1, 0).dot(reduced_data) / u.sum(axis=0)[:, np.newaxis]
        
        # Periksa konvergensi
        if np.linalg.norm(centroids_new - centroids) < error:
            break
        
        centroids = centroids_new
    
    # Assign cluster labels
    data['Cluster_FCM'] = np.argmax(u, axis=1)
    
    return data

# Visualize clusters
def visualize_clusters(data, method):
    plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data[method], cmap='rainbow')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title(f'{method} Clustering')

# Load data
file_name = 'Mall_Customers.csv'
data = load_data(file_name)

# Allow user to input the number of clusters
num_clusters_fcm = int(input("Enter the number of clusters: "))

# Perform clustering using Fuzzy C-Means
clustered_data_fcm = fuzzy_cmeans_clustering(data.copy(), num_clusters_fcm)

# Display histograms for males and females
males_age = clustered_data_fcm[clustered_data_fcm['Gender'] == 'Male']['Age']
females_age = clustered_data_fcm[clustered_data_fcm['Gender'] == 'Female']['Age']

# Males histogram
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
age_bins = range(15, 75, 5)
sns.histplot(males_age, bins=age_bins, kde=False, color='#0066ff', ax=ax1, edgecolor="k", linewidth=2)
ax1.set_xticks(age_bins)
ax1.set_ylim(top=25)
ax1.set_title('Males')
ax1.set_ylabel('Count')
ax1.text(45, 23, "TOTAL count: {}".format(males_age.count()))
ax1.text(45, 22, "Mean age: {:.1f}".format(males_age.mean()))

# Females histogram
sns.histplot(females_age, bins=age_bins, kde=False, color='#cc66ff', ax=ax2, edgecolor="k", linewidth=2)
ax2.set_xticks(age_bins)
ax2.set_title('Females')
ax2.set_ylabel('Count')
ax2.text(45, 23, "TOTAL count: {}".format(females_age.count()))
ax2.text(45, 22, "Mean age: {:.1f}".format(females_age.mean()))

# Show histograms
plt.show()

# Visualize results
visualize_clusters(clustered_data_fcm, 'Cluster_FCM')
plt.show()

# Display a random sample of 10 rows from the dataset
display(clustered_data_fcm.sample(10))
