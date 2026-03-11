import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
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

st.set_page_config(page_title="Black Friday Sales Insights", page_icon="🛍️", layout="wide")
st.title("🛍️ Mining the Future: Black Friday Sales Insights")
st.markdown("**InsightMart Analytics | Data Mining Summative Project**")
st.markdown("---")

st.sidebar.header("📂 Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload train.csv (Black Friday dataset)", type="csv")

if uploaded is None:
    st.info("👆 Please upload your Black Friday dataset (train.csv) using the sidebar to begin.")
    st.markdown("**Download:** [Google Drive Dataset Link](https://drive.google.com/drive/folders/13DxtCVj3S_AAYXG5THw2mmr6_VA1N3L9)")
    st.stop()

@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)
    df["Product_Category_2"] = df["Product_Category_2"].fillna(0).astype(int)
    df["Product_Category_3"] = df["Product_Category_3"].fillna(0).astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Gender_Encoded"] = df["Gender"].map({"M": 0, "F": 1})
    age_map = {"0-17": 1, "18-25": 2, "26-35": 3, "36-45": 4, "46-50": 5, "51-55": 6, "55+": 7}
    df["Age_Encoded"] = df["Age"].map(age_map)
    df["City_Encoded"] = df["City_Category"].map({"A": 1, "B": 2, "C": 3})
    df["Stay_Encoded"] = df["Stay_In_Current_City_Years"].replace("4+", 4).astype(int)
    df["Purchase_Normalized"] = StandardScaler().fit_transform(df[["Purchase"]])
    df["Z_Score"] = np.abs(stats.zscore(df["Purchase"]))
    df["Is_Anomaly"] = df["Z_Score"] > 3
    return df

try:
    df = load_and_clean(uploaded)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.sidebar.success(f"✅ Loaded {len(df):,} transactions")

@st.cache_data
def run_clustering(_df):
    user_df = _df.groupby("User_ID").agg(
        Age_Encoded=("Age_Encoded", "first"),
        Gender_Encoded=("Gender_Encoded", "first"),
        Occupation=("Occupation", "first"),
        Marital_Status=("Marital_Status", "first"),
        Avg_Purchase=("Purchase", "mean"),
        Total_Purchase=("Purchase", "sum"),
        Num_Transactions=("Purchase", "count")
    ).reset_index()
    features = ["Age_Encoded","Gender_Encoded","Occupation","Marital_Status","Avg_Purchase","Num_Transactions"]
    X = StandardScaler().fit_transform(user_df[features])
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in range(2, 11)]
    user_df["Cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    cm = user_df.groupby("Cluster")["Avg_Purchase"].mean().sort_values()
    lmap = {cm.index[0]:"Budget Shoppers", cm.index[1]:"Casual Buyers", cm.index[2]:"Regular Spenders", cm.index[3]:"Premium Buyers"}
    user_df["Cluster_Label"] = user_df["Cluster"].map(lmap)
    Xp = PCA(n_components=2).fit_transform(X)
    user_df["PCA1"] = Xp[:, 0]
    user_df["PCA2"] = Xp[:, 1]
    return user_df, inertias

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 EDA","👥 Clustering","🔗 Association Rules","🚨 Anomaly Detection","💡 Insights"])

with tab1:
    st.header("Exploratory Data Analysis")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Customers", f"{df['User_ID'].nunique():,}")
    c3.metric("Products", f"{df['Product_ID'].nunique():,}")
    c4.metric("Avg Purchase", f"${df['Purchase'].mean():,.0f}")
    st.markdown("---")

    ca, cb = st.columns(2)
    with ca:
        st.subheader("Avg Purchase by Age Group")
        age_order = ["0-17","18-25","26-35","36-45","46-50","51-55","55+"]
        age_data = df.groupby("Age")["Purchase"].mean().reindex(age_order).dropna()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(age_data.index, age_data.values, color="steelblue")
        ax.set_ylabel("Avg Purchase (USD)"); plt.xticks(rotation=30); plt.tight_layout()
        st.pyplot(fig); plt.close()
    with cb:
        st.subheader("Avg Purchase by Gender")
        gd = df.groupby("Gender")["Purchase"].mean()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(["Male","Female"], [gd.get("M",0), gd.get("F",0)], color=["royalblue","hotpink"])
        ax.set_ylabel("Avg Purchase (USD)"); plt.tight_layout()
        st.pyplot(fig); plt.close()

    cc, cd = st.columns(2)
    with cc:
        st.subheader("Top 10 Product Categories")
        tc = df["Product_Category_1"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(tc.index.astype(str), tc.values, color="coral")
        ax.set_xlabel("Category"); ax.set_ylabel("Count"); plt.tight_layout()
        st.pyplot(fig); plt.close()
    with cd:
        st.subheader("Avg Purchase per Category")
        ac = df.groupby("Product_Category_1")["Purchase"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.barh(ac.index.astype(str), ac.values, color="mediumseagreen")
        ax.set_xlabel("Avg Purchase (USD)"); plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.subheader("Correlation Heatmap")
    num_cols = ["Age_Encoded","Gender_Encoded","Occupation","City_Encoded","Stay_Encoded",
                "Marital_Status","Product_Category_1","Product_Category_2","Product_Category_3","Purchase"]
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with tab2:
    st.header("Customer Clustering")
    try:
        user_df, inertias = run_clustering(df)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Elbow Method")
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(range(2,11), inertias, "bo-")
            ax.axvline(x=4, color="red", linestyle="--", label="k=4")
            ax.set_xlabel("Clusters"); ax.set_ylabel("Inertia"); ax.legend(); plt.tight_layout()
            st.pyplot(fig); plt.close()
        with c2:
            st.subheader("Segment Distribution")
            counts = user_df["Cluster_Label"].value_counts()
            fig, ax = plt.subplots(figsize=(6,4))
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                   colors=["#4e79a7","#f28e2b","#e15759","#76b7b2"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("PCA Cluster Scatter")
        COLORS = {"Budget Shoppers":"#4e79a7","Casual Buyers":"#f28e2b","Regular Spenders":"#e15759","Premium Buyers":"#76b7b2"}
        fig, ax = plt.subplots(figsize=(10,6))
        for lbl, grp in user_df.groupby("Cluster_Label"):
            ax.scatter(grp["PCA1"], grp["PCA2"], label=lbl, alpha=0.5, s=15, color=COLORS.get(lbl,"grey"))
        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Cluster Statistics")
        st.dataframe(user_df.groupby("Cluster_Label").agg(
            Avg_Purchase=("Avg_Purchase","mean"), Total_Purchase=("Total_Purchase","mean"),
            Num_Transactions=("Num_Transactions","mean"), Count=("User_ID","count")
        ).round(2))
    except Exception as e:
        st.error(f"Clustering error: {e}")

with tab3:
    st.header("Association Rule Mining")
    min_support = st.slider("Minimum Support", 0.05, 0.5, 0.10, 0.01)
    min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.2, 0.1)

    @st.cache_data
    def run_apriori(_df, support, lift):
        basket = _df.groupby("User_ID").apply(
            lambda x: list(set(
                ["Cat1_"+str(int(c)) for c in x["Product_Category_1"] if c!=0]+
                ["Cat2_"+str(int(c)) for c in x["Product_Category_2"] if c!=0]+
                ["Cat3_"+str(int(c)) for c in x["Product_Category_3"] if c!=0]
            ))
        ).tolist()
        te = TransactionEncoder()
        bdf = pd.DataFrame(te.fit_transform(basket), columns=te.columns_)
        freq = apriori(bdf, min_support=support, use_colnames=True)
        if len(freq)==0: return pd.DataFrame()
        rules = association_rules(freq, metric="lift", min_threshold=lift, num_itemsets=len(freq))
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        return rules.sort_values("lift", ascending=False)

    try:
        rdf = run_apriori(df, min_support, min_lift)
        if rdf.empty:
            st.warning("No rules found. Lower the support or lift threshold.")
        else:
            st.success(f"✅ {len(rdf)} rules found")
            top = rdf.head(15).copy()
            top["rule"] = top["antecedents"]+" → "+top["consequents"]
            fig, ax = plt.subplots(figsize=(10,6))
            ax.barh(top["rule"], top["lift"], color="darkorange")
            ax.axvline(x=1, color="red", linestyle="--"); ax.set_xlabel("Lift")
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(rdf[["antecedents","consequents","support","confidence","lift"]].round(4))
    except Exception as e:
        st.error(f"Association rules error: {e}")

with tab4:
    st.header("Anomaly Detection")
    z_thresh = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
    anom = df[df["Z_Score"] > z_thresh]
    c1,c2,c3 = st.columns(3)
    c1.metric("Anomalies", f"{len(anom):,}")
    c2.metric("Max Purchase", f"${anom['Purchase'].max():,.0f}" if len(anom) else "N/A")
    c3.metric("Rate", f"{len(anom)/len(df)*100:.2f}%")

    ca, cb = st.columns(2)
    with ca:
        norm = df[df["Z_Score"] <= z_thresh]
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(range(min(5000,len(norm))), norm["Purchase"].iloc[:5000], alpha=0.3, s=5, label="Normal", color="steelblue")
        if len(anom):
            ax.scatter(range(len(anom)), anom["Purchase"], alpha=0.9, s=25, label="Anomaly", color="red")
        ax.set_ylabel("Purchase"); ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()
    with cb:
        Q1,Q3 = df["Purchase"].quantile(0.25), df["Purchase"].quantile(0.75)
        upper = Q3+1.5*(Q3-Q1)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(df["Purchase"], bins=60, color="steelblue", alpha=0.7)
        ax.axvline(upper, color="red", linestyle="--", label=f"IQR Upper")
        ax.set_xlabel("Purchase"); ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

    if len(anom):
        st.subheader("Anomalies by Age")
        st.dataframe(anom.groupby("Age")["Purchase"].agg(["mean","count"]).reset_index())
        st.subheader("Sample Anomalous Transactions")
        st.dataframe(anom[["User_ID","Age","Gender","Occupation","City_Category","Purchase","Z_Score"]].head(20))

with tab5:
    st.header("💡 Key Insights & Recommendations")
    try:
        age_spend = df.groupby("Age")["Purchase"].mean().sort_values(ascending=False)
        gender_spend = df.groupby("Gender")["Purchase"].mean()
        top_cat = df["Product_Category_1"].value_counts().idxmax()
        city_spend = df.groupby("City_Category")["Purchase"].mean().sort_values(ascending=False)
        udf, _ = run_clustering(df)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🏆 Top Spending Age Group")
            st.success(f"**{age_spend.index[0]}** — avg ${age_spend.iloc[0]:,.0f}")
            st.markdown("### 👔 Gender Spending")
            st.info(f"Male: **${gender_spend.get('M',0):,.0f}** | Female: **${gender_spend.get('F',0):,.0f}**")
            st.markdown("### 🏙️ Top City")
            st.info(f"City **{city_spend.index[0]}** — avg ${city_spend.iloc[0]:,.0f}")
        with c2:
            st.markdown("### 🛒 Most Popular Category")
            st.success(f"Product Category **{top_cat}**")
            st.markdown("### 🚨 Anomalies")
            n = df["Is_Anomaly"].sum()
            st.warning(f"**{n:,}** high-spenders ({n/len(df)*100:.2f}%)")
            st.markdown("### 👥 Segments")
            for lbl, cnt in udf["Cluster_Label"].value_counts().items():
                st.write(f"- **{lbl}**: {cnt:,} customers")

        st.markdown("---")
        st.markdown("### 📋 Recommendations")
        st.markdown("""
        1. **Target 26–35 age group** with premium bundles — highest avg spend.
        2. **Male shoppers** drive more volume — tailor promotions accordingly.
        3. **City A customers** have higher capacity — focus loyalty programs there.
        4. **Flag anomalous spenders** — potential bulk buyers or fraud.
        5. **Cross-sell via association rules** — bundle co-purchased categories.
        6. **Engage Budget Shoppers** with discounts to move them up the ladder.
        """)
    except Exception as e:
        st.error(f"Insights error: {e}")

st.markdown("---")
st.caption("🎓 Data Mining Summative | InsightMart Analytics | Black Friday Dataset")
