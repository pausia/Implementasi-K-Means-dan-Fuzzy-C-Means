import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Load data
def load_data(file_name):
    full_path = r'C:\Users\ACER\Documents\A Semester 7\Logika Fuzzy\Pak Adha\Tugas-2\code\csv' + '\\' + file_name
    data = pd.read_csv(full_path)
    return data

# K-Means clustering
def kmeans_clustering(data, num_clusters):
    features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans = KMeans(n_clusters=num_clusters)
    data['Cluster_KMeans'] = kmeans.fit_predict(features)
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

# Input jumlah cluster dari pengguna
num_clusters_kmeans = int(input("Enter the number of clusters for K-Means: "))

# Perform clustering using KMeans
clustered_data_kmeans = kmeans_clustering(data.copy(), num_clusters_kmeans)

# Display histograms for males and females
males_age = clustered_data_kmeans[clustered_data_kmeans['Gender'] == 'Male']['Age']
females_age = clustered_data_kmeans[clustered_data_kmeans['Gender'] == 'Female']['Age']

age_bins = range(15, 75, 5)

# Males histogram
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.distplot(males_age, bins=age_bins, kde=False, color='#0066ff', ax=ax1, hist_kws=dict(edgecolor="k", linewidth=2))
ax1.set_xticks(age_bins)
ax1.set_ylim(top=25)
ax1.set_title('Males')
ax1.set_ylabel('Count')
ax1.text(45, 23, "TOTAL count: {}".format(males_age.count()))
ax1.text(45, 22, "Mean age: {:.1f}".format(males_age.mean()))

# Females histogram
sns.distplot(females_age, bins=age_bins, kde=False, color='#cc66ff', ax=ax2, hist_kws=dict(edgecolor="k", linewidth=2))
ax2.set_xticks(age_bins)
ax2.set_title('Females')
ax2.set_ylabel('Count')
ax2.text(45, 23, "TOTAL count: {}".format(females_age.count()))
ax2.text(45, 22, "Mean age: {:.1f}".format(females_age.mean()))

# Show histograms
plt.show()

# Visualize results
visualize_clusters(clustered_data_kmeans, 'Cluster_KMeans')
plt.show()

# Display a random sample of 10 rows from the dataset
display(clustered_data_kmeans.sample(10))
