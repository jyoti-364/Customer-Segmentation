"""
takes the customer feature table and does scaling + PCA + KMeans clustering
saves the elbow/silhouette plot, the cluster scatter plot, and the final results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

IN_PATH = "../data/customer_features.csv"
OUT_DIR = "../outputs"

FEATURE_COLS = ["Recency", "Frequency", "Monetary", "AvgOrderValue",
                 "UniqueProducts", "TotalItems", "TenureDays"]
SKEWED_COLS = ["Frequency", "Monetary", "AvgOrderValue", "UniqueProducts", "TotalItems"]


def scale_features(df):
    # spend/frequency data is really skewed - a few wholesale-ish customers spend
    # way more than everyone else and it throws off the clustering if you dont fix
    # this first (learned this the hard way, first attempt without log transform
    # gave a weird 17-customer "cluster" that was basically just outliers)
    df_t = df.copy()
    for col in SKEWED_COLS:
        df_t[col] = np.log1p(df_t[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_t[FEATURE_COLS])
    return X_scaled


def run_pca(X_scaled, n_components=3):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print('variance explained per component:', pca.explained_variance_ratio_)
    print('total variance explained:', sum(pca.explained_variance_ratio_))
    return X_pca


def elbow_and_silhouette(X_pca, k_range=range(2, 11)):
    inertia_list = []
    silhouette_list = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        cluster_labels = km.fit_predict(X_pca)
        inertia_list.append(km.inertia_)
        silhouette_list.append(silhouette_score(X_pca, cluster_labels))
        print(f'k={k}  inertia={km.inertia_:.0f}  silhouette={silhouette_list[-1]:.3f}')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(list(k_range), inertia_list, marker="o")
    ax[0].set_title("Elbow Plot")
    ax[0].set_xlabel("K")
    ax[0].set_ylabel("Inertia")

    ax[1].plot(list(k_range), silhouette_list, marker="o", color="orange")
    ax[1].set_title("Silhouette Score")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("Score")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/elbow_silhouette.png", dpi=150)
    print('saved elbow/silhouette plot')


def fit_final_kmeans(X_pca, k):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_pca)
    print(f'final model K={k}, silhouette score={silhouette_score(X_pca, labels):.3f}')
    return labels


def plot_clusters(X_pca, labels):
    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.6, s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Customer Clusters (PC1 vs PC2)")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cluster_scatter.png", dpi=150)
    print('saved cluster scatter plot')


def profile_clusters(df, labels):
    df = df.copy()
    df["Cluster"] = labels

    summary = df.groupby("Cluster")[FEATURE_COLS].mean().round(1)
    summary["num_customers"] = df.groupby("Cluster").size()
    summary["pct"] = (summary["num_customers"] / len(df) * 100).round(1)

    print(summary)

    summary.to_csv(f"{OUT_DIR}/cluster_profiles.csv")
    df.to_csv(f"{OUT_DIR}/customers_with_clusters.csv", index=False)
    print('saved cluster profiles and labeled customer data')
    return summary


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)

    X_scaled = scale_features(df)
    X_pca = run_pca(X_scaled, n_components=3)

    elbow_and_silhouette(X_pca)

    # going with K=4 - not the highest silhouette score (K=2 wins there) but K=2
    # is basically just splitting outliers from everyone else, not actually useful.
    # elbow flattens around here and silhouette is stable through K=4-6
    K = 4
    labels = fit_final_kmeans(X_pca, K)

    plot_clusters(X_pca, labels)
    profile_clusters(df, labels)
