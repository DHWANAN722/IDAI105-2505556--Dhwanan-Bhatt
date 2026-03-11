import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday Sales Insights",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Mining the Future: Black Friday Sales Insights")
st.markdown("**InsightMart Analytics | Data Mining Summative Project**")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload train.csv (Black Friday dataset)", type="csv")

if uploaded is None:
    st.info("👆 Please upload your Black Friday dataset (train.csv) using the sidebar to begin.")
    st.markdown("""
    **Download the dataset from:**  
    [Google Drive Dataset Link](https://drive.google.com/drive/folders/13DxtCVj3S_AAYXG5THw2mmr6_VA1N3L9)
    
    Then upload the CSV file using the sidebar panel on the left.
    """)
    st.stop()

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)

    # Fill missing
    df["Product_Category_2"] = df["Product_Category_2"].fillna(0).astype(int)
    df["Product_Category_3"] = df["Product_Category_3"].fillna(0).astype(int)
    df = df.drop_duplicates()

    # Encode
    df["Gender_Encoded"] = df["Gender"].map({"M": 0, "F": 1})
    age_map = {"0-17": 1, "18-25": 2, "26-35": 3, "36-45": 4, "46-50": 5, "51-55": 6, "55+": 7}
    df["Age_Encoded"] = df["Age"].map(age_map)
    df["City_Encoded"] = df["City_Category"].map({"A": 1, "B": 2, "C": 3})
    df["Stay_Encoded"] = df["Stay_In_Current_City_Years"].replace("4+", 4).astype(int)

    # Normalize
    scaler = StandardScaler()
    df["Purchase_Normalized"] = scaler.fit_transform(df[["Purchase"]])

    # Anomalies
    df["Z_Score"] = np.abs(stats.zscore(df["Purchase"]))
    df["Is_Anomaly"] = df["Z_Score"] > 3

    return df

df = load_and_clean(uploaded)
st.sidebar.success(f"✅ Loaded {len(df):,} transactions")

# ── Navigation tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "👥 Clustering", "🔗 Association Rules", "🚨 Anomaly Detection", "💡 Insights"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 – EDA
# ════════════════════════════════════════════════════════════════
with tab1:
    st.header("Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Unique Customers", f"{df['User_ID'].nunique():,}")
    col3.metric("Unique Products", f"{df['Product_ID'].nunique():,}")
    col4.metric("Avg Purchase", f"${df['Purchase'].mean():,.0f}")

    st.markdown("---")

    # Purchase by Age & Gender
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Purchase by Age Group")
        age_order = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
        age_data = df.groupby("Age")["Purchase"].mean().reindex(age_order)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(age_data.index, age_data.values, color="steelblue")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Avg Purchase (USD)")
        ax.set_title("Average Purchase by Age Group")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Purchase by Gender")
        gender_data = df.groupby("Gender")["Purchase"].mean()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Male", "Female"], [gender_data.get("M", 0), gender_data.get("F", 0)],
               color=["royalblue", "hotpink"])
        ax.set_ylabel("Avg Purchase (USD)")
        ax.set_title("Average Purchase by Gender")
        st.pyplot(fig)
        plt.close()

    # Product categories
    st.subheader("Top Product Categories")
    col_c, col_d = st.columns(2)

    with col_c:
        top_cats = df["Product_Category_1"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(top_cats.index.astype(str), top_cats.values, color="coral")
        ax.set_title("Top 10 Product Categories (Category 1)")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    with col_d:
        avg_by_cat = df.groupby("Product_Category_1")["Purchase"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(avg_by_cat.index.astype(str), avg_by_cat.values, color="mediumseagreen")
        ax.set_title("Avg Purchase by Product Category")
        ax.set_xlabel("Avg Purchase (USD)")
        st.pyplot(fig)
        plt.close()

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = ["Age_Encoded", "Gender_Encoded", "Occupation", "City_Encoded",
                    "Stay_Encoded", "Marital_Status", "Product_Category_1",
                    "Product_Category_2", "Product_Category_3", "Purchase"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════
# TAB 2 – CLUSTERING
# ════════════════════════════════════════════════════════════════
with tab2:
    st.header("Customer Clustering")

    @st.cache_data
    def run_clustering(df):
        user_df = df.groupby("User_ID").agg(
            Age_Encoded=("Age_Encoded", "first"),
            Gender_Encoded=("Gender_Encoded", "first"),
            Occupation=("Occupation", "first"),
            Marital_Status=("Marital_Status", "first"),
            Avg_Purchase=("Purchase", "mean"),
            Total_Purchase=("Purchase", "sum"),
            Num_Transactions=("Purchase", "count")
        ).reset_index()

        features = ["Age_Encoded", "Gender_Encoded", "Occupation",
                    "Marital_Status", "Avg_Purchase", "Num_Transactions"]
        X = StandardScaler().fit_transform(user_df[features])

        # Elbow
        inertias = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)

        # Final model
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        user_df["Cluster"] = kmeans.fit_predict(X)

        cluster_means = user_df.groupby("Cluster")["Avg_Purchase"].mean().sort_values()
        labels = {
            cluster_means.index[0]: "Budget Shoppers",
            cluster_means.index[1]: "Casual Buyers",
            cluster_means.index[2]: "Regular Spenders",
            cluster_means.index[3]: "Premium Buyers"
        }
        user_df["Cluster_Label"] = user_df["Cluster"].map(labels)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        user_df["PCA1"] = X_pca[:, 0]
        user_df["PCA2"] = X_pca[:, 1]

        return user_df, inertias

    user_df, inertias = run_clustering(df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Elbow Method")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(2, 11), inertias, "bo-")
        ax.set_title("Elbow Curve – Choosing Optimal k")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Inertia")
        ax.axvline(x=4, color="red", linestyle="--", label="Chosen k=4")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Cluster Distribution")
        counts = user_df["Cluster_Label"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
        ax.set_title("Customer Segment Distribution")
        st.pyplot(fig)
        plt.close()

    st.subheader("Cluster Scatter Plot (PCA)")
    colors = {"Budget Shoppers": "#4e79a7", "Casual Buyers": "#f28e2b",
              "Regular Spenders": "#e15759", "Premium Buyers": "#76b7b2"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, grp in user_df.groupby("Cluster_Label"):
        ax.scatter(grp["PCA1"], grp["PCA2"], label=label,
                   alpha=0.5, s=15, color=colors[label])
    ax.set_title("Customer Segments (PCA Projection)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.subheader("Cluster Statistics")
    cluster_stats = user_df.groupby("Cluster_Label").agg(
        Avg_Purchase=("Avg_Purchase", "mean"),
        Total_Purchase=("Total_Purchase", "mean"),
        Num_Transactions=("Num_Transactions", "mean"),
        Count=("User_ID", "count")
    ).round(2)
    st.dataframe(cluster_stats, width="stretch")

# ════════════════════════════════════════════════════════════════
# TAB 3 – ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════
with tab3:
    st.header("Association Rule Mining")

    min_support = st.slider("Minimum Support", 0.05, 0.5, 0.10, 0.01)
    min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.2, 0.1)

    @st.cache_data
    def run_apriori(df, support, lift):
        basket = df.groupby("User_ID").apply(
            lambda x: list(set(
                ["Cat1_" + str(int(c)) for c in x["Product_Category_1"] if c != 0] +
                ["Cat2_" + str(int(c)) for c in x["Product_Category_2"] if c != 0] +
                ["Cat3_" + str(int(c)) for c in x["Product_Category_3"] if c != 0]
            ))
        ).tolist()
        te = TransactionEncoder()
        basket_df = pd.DataFrame(te.fit_transform(basket), columns=te.columns_)
        freq = apriori(basket_df, min_support=support, use_colnames=True)
        if len(freq) == 0:
            return pd.DataFrame()
        rules = association_rules(freq, metric="lift", min_threshold=lift)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        return rules.sort_values("lift", ascending=False)

    rules_df = run_apriori(df, min_support, min_lift)

    if rules_df.empty:
        st.warning("No rules found with current thresholds. Try lowering support or lift.")
    else:
        st.success(f"✅ Found {len(rules_df)} association rules")

        st.subheader("Top Rules by Lift")
        top_rules = rules_df.head(15)
        top_rules["rule"] = top_rules["antecedents"] + " → " + top_rules["consequents"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_rules["rule"], top_rules["lift"], color="darkorange")
        ax.axvline(x=1, color="red", linestyle="--", label="Lift = 1")
        ax.set_title("Top 15 Association Rules by Lift")
        ax.set_xlabel("Lift")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("All Rules Table")
        st.dataframe(
            rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].round(4),
            width="stretch"
        )

# ════════════════════════════════════════════════════════════════
# TAB 4 – ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════
with tab4:
    st.header("Anomaly Detection")

    z_threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
    anomalies = df[df["Z_Score"] > z_threshold]

    col1, col2, col3 = st.columns(3)
    col1.metric("Anomalies Detected", f"{len(anomalies):,}")
    col2.metric("Max Anomalous Purchase", f"${anomalies['Purchase'].max():,.0f}" if len(anomalies) else "N/A")
    col3.metric("Anomaly Rate", f"{len(anomalies)/len(df)*100:.2f}%")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Normal vs Anomalous Purchases")
        normal = df[~(df["Z_Score"] > z_threshold)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(range(min(5000, len(normal))), normal["Purchase"].head(5000),
                   alpha=0.3, s=5, label="Normal", color="steelblue")
        ax.scatter(range(len(anomalies)), anomalies["Purchase"],
                   alpha=0.9, s=25, label="Anomaly", color="red")
        ax.set_title("Purchase Anomalies")
        ax.set_ylabel("Purchase (USD)")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Purchase Distribution")
        Q1 = df["Purchase"].quantile(0.25)
        Q3 = df["Purchase"].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["Purchase"], bins=60, color="steelblue", alpha=0.7)
        ax.axvline(upper, color="red", linestyle="--", label=f"IQR Upper ({upper:,.0f})")
        ax.set_title("Purchase Distribution with IQR Bound")
        ax.set_xlabel("Purchase (USD)")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    if len(anomalies) > 0:
        st.subheader("Anomalous Spenders by Age Group")
        age_anom = anomalies.groupby("Age")["Purchase"].agg(["mean", "count"]).reset_index()
        age_anom.columns = ["Age Group", "Avg Purchase", "Count"]
        st.dataframe(age_anom, width="stretch")

        st.subheader("Sample Anomalous Transactions")
        st.dataframe(anomalies[["User_ID", "Age", "Gender", "Occupation",
                                 "City_Category", "Purchase", "Z_Score"]].head(20),
                     width="stretch")

# ════════════════════════════════════════════════════════════════
# TAB 5 – INSIGHTS
# ════════════════════════════════════════════════════════════════
with tab5:
    st.header("💡 Key Insights & Recommendations")

    age_spend = df.groupby("Age")["Purchase"].mean().sort_values(ascending=False)
    gender_spend = df.groupby("Gender")["Purchase"].mean()
    top_cat = df["Product_Category_1"].value_counts().idxmax()
    city_spend = df.groupby("City_Category")["Purchase"].mean().sort_values(ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Top Spending Age Group")
        st.success(f"**{age_spend.index[0]}** with avg purchase of **${age_spend.iloc[0]:,.0f}**")

        st.markdown("### 👔 Gender Spending")
        m_avg = gender_spend.get("M", 0)
        f_avg = gender_spend.get("F", 0)
        higher = "Males" if m_avg > f_avg else "Females"
        st.info(f"**{higher}** spend more on average — Male: ${m_avg:,.0f} | Female: ${f_avg:,.0f}")

        st.markdown("### 🏙️ City Category")
        st.info(f"City **{city_spend.index[0]}** has the highest average spend: **${city_spend.iloc[0]:,.0f}**")

    with col2:
        st.markdown("### 🛒 Most Popular Product Category")
        st.success(f"Product Category **{top_cat}** dominates purchases")

        st.markdown("### 🚨 Anomalies")
        n_anom = df["Is_Anomaly"].sum()
        pct = n_anom / len(df) * 100
        st.warning(f"**{n_anom:,}** unusual high-spenders detected (**{pct:.2f}%** of transactions)")

        st.markdown("### 👥 Customer Segments")
        user_df_insight, _ = run_clustering(df)
        counts = user_df_insight["Cluster_Label"].value_counts()
        for label, count in counts.items():
            st.write(f"- **{label}**: {count:,} customers")

    st.markdown("---")
    st.markdown("### 📋 Recommendations")
    st.markdown("""
    1. **Target 26–35 age group** with premium product bundles — they are the highest spenders.
    2. **Male shoppers** drive more revenue — tailor promotions to male-dominant categories.
    3. **City A customers** have higher spend capacity — focus loyalty programs there.
    4. **Monitor anomalous spenders** — they may be bulk buyers or potential fraud; flag for review.
    5. **Cross-sell using association rules** — bundle frequently co-purchased categories for combo deals.
    6. **Engage Budget Shoppers** with discount campaigns to move them up the spending ladder.
    """)

st.markdown("---")
st.caption("🎓 Data Mining Summative Project | InsightMart Analytics | Black Friday Dataset")
