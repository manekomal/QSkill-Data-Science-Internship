# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 2. Load Dataset
df = pd.read_excel("Task_3/Online Retail.xlsx")

# 3. Clean Data
df = df.drop_duplicates()

df = df.dropna(subset=["CustomerID"])

df["InvoiceNo"] = df["InvoiceNo"].astype("string")

df = df[~df["InvoiceNo"].str.startswith("C", na=False)
]

df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
]

# 4. Calculate RFM
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

reference_date = (
    df["InvoiceDate"].max() + pd.Timedelta(days=1)
)

df["TotalAmount"] = (df["Quantity"] * df["UnitPrice"])

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (
        reference_date - x.max()
    ).days,
    "InvoiceNo": "nunique", "TotalAmount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm = rfm.reset_index()

# 5. Analyze Purchasing behavior
print("Average Recency:", round(rfm["Recency"].mean(), 2))
print("Average Frequency:", round(rfm["Frequency"].mean(), 2))
print("Average Monetary:", round(rfm["Monetary"].mean(), 2))

# 6. Visualize Purchasing Patterns
plt.figure(figsize=(8, 5))

sns.histplot(rfm["Recency"], bins=30)
plt.xlabel("Recency")
plt.ylabel("Number of Customers")
plt.title("Customer Recency Distribution")
plt.savefig("Task_3/recency_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(rfm["Frequency"], bins=30)
plt.xlabel("Frequency")
plt.ylabel("Number of Customers")
plt.title("Customer Frequency Distribution")
plt.savefig("Task_3/frequency_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(rfm["Monetary"], bins=30)
plt.xlabel("Monetary Value")
plt.ylabel("Number of Customers")
plt.title("Customer Monetary Value Distribution")
plt.savefig("Task_3/monetary_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x="Frequency", y="Monetary")
plt.xlabel("Frequency")
plt.ylabel("Monetary Value")
plt.title("Frequency vs Monetary Value")
plt.savefig("Task_3/frequency_vs_monetary.png", dpi=300, bbox_inches="tight")
plt.show()

# 7. Prepare Data For Clustering
rfm_clustering = rfm[
    ["Recency", "Frequency", "Monetary"]
].copy()

# 8. Determine Suitable Clusters
rfm_clustering = rfm_clustering.apply(np.log1p)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_clustering)

inertia = []
silhouette_scores = []

k_values = range(2, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=43,
        n_init=10
    )

    labels = model.fit_predict(rfm_scaled)

    inertia.append(model.inertia_)

    silhouette_scores.append(
        silhouette_score(
            rfm_scaled,
            labels
        )
    )

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker="o", linewidth=2)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("Task_3/elbow_method.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker="o", linewidth=2)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.savefig("Task_3/silhouette_score.png", dpi=300, bbox_inches="tight")
plt.show()

# 9. Apply K-Means Clustering
kmeans = KMeans(
    n_clusters=2,
    random_state=43,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    rfm_scaled
)

# 10. Analyze Customer Segments
segment_summary = rfm.groupby("Cluster").agg(
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean"),
    Customer_Count=("CustomerID", "count")
)

print("\nCustomer Segment Summary:")
print(segment_summary.round(2))

# 11. Describe Segment Characterisrics
overall_recency = rfm["Recency"].median()
overall_frequency = rfm["Frequency"].median()
overall_monetary = rfm["Monetary"].median()

for cluster in sorted(rfm["Cluster"].unique()):
    segment = segment_summary.loc[cluster]

    recency = segment["Recency"]
    frequency = segment["Frequency"]
    monetary = segment["Monetary"]

    print("\nCluster:", cluster)

    if(
        recency <= overall_recency
        and frequency >= overall_frequency
        and monetary >= overall_monetary
    ):
        print("Characteristics: High-value and active customers.")

    elif (
        recency > overall_recency
        and monetary >= overall_monetary
    ):
        print(
            "Characteristics: Valuable customers with lower recent activity."
        )

    elif (
        recency <= overall_recency
        and frequency >= overall_frequency
    ):
        print("Characteristics: Regular and loyal customers.")

    elif (   
        recency > overall_recency
        and frequency < overall_frequency
        and monetary < overall_monetary
    ):
        print("Characteristics: Low-frequency and low-value customers.")

    else:
        print("Characteristics: Moderate-value customer segment.") 

# 12. Business Recommendations
for cluster in sorted(rfm["Cluster"].unique()):

    segment = segment_summary.loc[cluster]

    recency = segment["Recency"]
    frequency = segment["Frequency"]
    monetary = segment["Monetary"]

    print("\nCluster:", cluster)

    if (
        recency <= overall_recency
        and frequency >= overall_frequency
        and monetary >= overall_monetary
    ):
        print(
            "Recommendation: Loyalty rewards and personalized offers."
        )

    elif (
        recency > overall_recency
        and monetary >= overall_monetary
    ):
        print(
            "Recommendation: Re-engagement campaigns and special discounts."
        )

    elif (
        recency <= overall_recency
        and frequency >= overall_frequency
    ):
        print(
            "Recommendation: Loyalty programs and repeat-purchase offers."
        )

    elif (
        recency > overall_recency
        and frequency < overall_frequency
        and monetary < overall_monetary
    ):
        print(
            "Recommendation: Targeted offers and introductory discounts."
        )

    else:
        print(
            "Recommendation: Personalized offers and cross-selling."
        )
   