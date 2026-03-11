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

st.set_page_config(page_title="⚡ Black Friday Intelligence", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0a0010 !important;
    color: #e0e0ff !important;
}
[data-testid="stSidebar"] {
    background: #0d0020 !important;
    border-right: 2px solid #7b2fff !important;
    box-shadow: 4px 0 20px #7b2fff55 !important;
}
h1 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 900 !important;
    font-size: 2.2rem !important;
    color: #ffffff !important;
    text-shadow: 0 0 10px #bf5fff, 0 0 25px #7b2fff, 0 0 50px #5500cc !important;
    letter-spacing: 3px !important;
}
h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: #d0a0ff !important;
    text-shadow: 0 0 12px #9b4fff, 0 0 25px #6600cc !important;
    letter-spacing: 2px !important;
}
p, li, label, span { color: #d0d0f0 !important; font-family: 'Share Tech Mono', monospace !important; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a0030 0%, #0d001a 100%) !important;
    border: 1px solid #7b2fff !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: 0 0 15px #7b2fff44, inset 0 0 15px #7b2fff11 !important;
}
[data-testid="stMetricValue"] {
    color: #bf5fff !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
    text-shadow: 0 0 10px #bf5fff !important;
}
[data-testid="stMetricLabel"] {
    color: #9090cc !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stTabs"] button {
    font-family: 'Orbitron', monospace !important;
    color: #9090cc !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #bf5fff !important;
    border-bottom: 2px solid #bf5fff !important;
    text-shadow: 0 0 10px #bf5fff !important;
}
hr { border-color: #7b2fff44 !important; box-shadow: 0 0 6px #7b2fff !important; }
[data-testid="stSidebar"] * { color: #c0a0ff !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0010; }
::-webkit-scrollbar-thumb { background: #7b2fff; border-radius: 3px; }
.neon-card {
    background: linear-gradient(135deg, #1a0030 0%, #0a0018 100%);
    border: 1px solid #7b2fff;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    box-shadow: 0 0 20px #7b2fff33, inset 0 0 20px #7b2fff0a;
}
</style>
""", unsafe_allow_html=True)

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor': '#0a0010', 'axes.facecolor': '#0d0020',
        'axes.edgecolor': '#7b2fff', 'axes.labelcolor': '#c0a0ff',
        'axes.titlecolor': '#d0b0ff', 'axes.grid': True,
        'grid.color': '#7b2fff', 'grid.alpha': 0.2,
        'xtick.color': '#c0a0ff', 'ytick.color': '#c0a0ff',
        'text.color': '#c0a0ff', 'legend.facecolor': '#0d0020',
        'legend.edgecolor': '#7b2fff', 'legend.labelcolor': '#c0a0ff',
    })

NEON = ["#bf5fff", "#00ff88", "#00eeff", "#ff8800", "#ff2266", "#ffee00"]

st.markdown("""
<div style='text-align:center; padding:10px 0 5px 0;'>
  <h1>⚡ BLACK FRIDAY INTELLIGENCE ⚡</h1>
  <p style='color:#9060cc; font-family:Share Tech Mono; letter-spacing:3px; font-size:0.85rem;'>
    🛍️ INSIGHTMART ANALYTICS &nbsp;|&nbsp; 📊 DATA MINING PROJECT &nbsp;|&nbsp; 🔬 AI MODULE
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.markdown("<p style='font-family:Orbitron;font-size:1rem;color:#bf5fff;text-shadow:0 0 10px #bf5fff;letter-spacing:2px;text-align:center;'>⚡ CONTROL PANEL</p>", unsafe_allow_html=True)
st.sidebar.header("📂 Upload Dataset")
uploaded = st.sidebar.file_uploader("Drop train.csv here", type="csv")

if uploaded is None:
    st.markdown("""
    <div class='neon-card' style='text-align:center;padding:40px;'>
      <p style='font-size:3rem;margin:0;'>🛍️</p>
      <p style='font-family:Orbitron;color:#bf5fff;font-size:1.1rem;text-shadow:0 0 10px #bf5fff;letter-spacing:2px;'>AWAITING DATA UPLOAD</p>
      <p style='color:#9090cc;font-size:0.85rem;'>👆 Upload your <b>train.csv</b> Black Friday dataset via the sidebar to begin analysis</p>
      <br>
      <p style='color:#7070aa;font-size:0.8rem;'>📥 Download from: <a href='https://drive.google.com/drive/folders/13DxtCVj3S_AAYXG5THw2mmr6_VA1N3L9' style='color:#bf5fff;'>🔗 Google Drive Link</a></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)
    df["Product_Category_2"] = df["Product_Category_2"].fillna(0).astype(int)
    df["Product_Category_3"] = df["Product_Category_3"].fillna(0).astype(int)
    df = df.drop_duplicates()
    df["Gender_Encoded"] = df["Gender"].map({"M": 0, "F": 1})
    age_map = {"0-17": 1, "18-25": 2, "26-35": 3, "36-45": 4, "46-50": 5, "51-55": 6, "55+": 7}
    df["Age_Encoded"] = df["Age"].map(age_map)
    df["City_Encoded"] = df["City_Category"].map({"A": 1, "B": 2, "C": 3})
    df["Stay_Encoded"] = df["Stay_In_Current_City_Years"].replace("4+", 4).astype(int)
    scaler = StandardScaler()
    df["Purchase_Normalized"] = scaler.fit_transform(df[["Purchase"]])
    df["Z_Score"] = np.abs(stats.zscore(df["Purchase"]))
    df["Is_Anomaly"] = df["Z_Score"] > 3
    return df

df = load_and_clean(uploaded)
st.sidebar.success(f"✅ {len(df):,} transactions loaded!")
st.sidebar.markdown(f"""
<div class='neon-card' style='margin-top:10px;font-size:0.78rem;'>
  👤 <b>{df['User_ID'].nunique():,}</b> unique customers<br>
  📦 <b>{df['Product_ID'].nunique():,}</b> unique products<br>
  💰 Avg purchase: <b>${df['Purchase'].mean():,.0f}</b>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊  EDA", "👥  Clustering", "🔗  Association Rules", "🚨  Anomaly Detection", "💡  Insights"])

# ── TAB 1: EDA ──
with tab1:
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown("*Uncovering patterns hidden in 550K+ Black Friday transactions*")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🧾 Transactions", f"{len(df):,}")
    c2.metric("👤 Customers", f"{df['User_ID'].nunique():,}")
    c3.metric("📦 Products", f"{df['Product_ID'].nunique():,}")
    c4.metric("💰 Avg Purchase", f"${df['Purchase'].mean():,.0f}")
    st.markdown("---")
    set_dark_style()

    ca, cb = st.columns(2)
    with ca:
        st.markdown("### 🎂 Purchase by Age Group")
        age_order = ["0-17","18-25","26-35","36-45","46-50","51-55","55+"]
        age_data = df.groupby("Age")["Purchase"].mean().reindex(age_order)
        fig, ax = plt.subplots(figsize=(7,4))
        bars = ax.bar(age_data.index, age_data.values, color=NEON[:len(age_data)], edgecolor="#bf5fff", linewidth=0.8, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+20, f"${h:,.0f}", ha="center", va="bottom", fontsize=7, color="#c0a0ff")
        ax.set_xlabel("Age Group"); ax.set_ylabel("Avg Purchase (USD)"); ax.set_title("💰 Average Purchase by Age Group")
        plt.xticks(rotation=30); fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()

    with cb:
        st.markdown("### ⚧ Purchase by Gender")
        gd = df.groupby("Gender")["Purchase"].mean()
        fig, ax = plt.subplots(figsize=(5,4))
        vals = [gd.get("M",0), gd.get("F",0)]
        bars = ax.bar(["👨 Male","👩 Female"], vals, color=["#00eeff","#ff2266"], edgecolor="#bf5fff", linewidth=1.2, width=0.5)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+20, f"${v:,.0f}", ha="center", va="bottom", fontsize=9, color="#c0a0ff")
        ax.set_ylabel("Avg Purchase (USD)"); ax.set_title("⚧ Gender Spending")
        fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()

    st.markdown("### 🏷️ Top Product Categories")
    cc, cd = st.columns(2)
    with cc:
        tc = df["Product_Category_1"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(tc.index.astype(str), tc.values, color=NEON*2, edgecolor="#bf5fff", linewidth=0.6, alpha=0.85)
        ax.set_title("🏆 Top 10 Most Purchased Categories"); ax.set_xlabel("Category ID"); ax.set_ylabel("Count")
        fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()
    with cd:
        abc = df.groupby("Product_Category_1")["Purchase"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.barh(abc.index.astype(str), abc.values, color="#00ff88", edgecolor="#bf5fff", linewidth=0.6, alpha=0.85)
        ax.set_title("💎 Avg Spend per Category"); ax.set_xlabel("Avg Purchase (USD)")
        fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()

    st.markdown("### 🌡️ Correlation Heatmap")
    numeric_cols = ["Age_Encoded","Gender_Encoded","Occupation","City_Encoded","Stay_Encoded","Marital_Status","Product_Category_1","Product_Category_2","Product_Category_3","Purchase"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(11,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="PuBu", ax=ax, linewidths=0.5, linecolor="#7b2fff44", annot_kws={"size":8,"color":"#e0e0ff"})
    ax.set_title("🌡️ Feature Correlation Matrix"); fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()

# ── TAB 2: CLUSTERING ──
with tab2:
    st.markdown("## 👥 Customer Segmentation")
    st.markdown("*Grouping shoppers by behaviour using K-Means clustering*")

    @st.cache_data
    def run_clustering(df):
        udf = df.groupby("User_ID").agg(
            Age_Encoded=("Age_Encoded","first"), Gender_Encoded=("Gender_Encoded","first"),
            Occupation=("Occupation","first"), Marital_Status=("Marital_Status","first"),
            Avg_Purchase=("Purchase","mean"), Total_Purchase=("Purchase","sum"),
            Num_Transactions=("Purchase","count")
        ).reset_index()
        features = ["Age_Encoded","Gender_Encoded","Occupation","Marital_Status","Avg_Purchase","Num_Transactions"]
        X = StandardScaler().fit_transform(udf[features])
        inertias = [KMeans(n_clusters=k,random_state=42,n_init=10).fit(X).inertia_ for k in range(2,11)]
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        udf["Cluster"] = kmeans.fit_predict(X)
        cm = udf.groupby("Cluster")["Avg_Purchase"].mean().sort_values()
        labels = {cm.index[0]:"💸 Budget Shoppers",cm.index[1]:"🛒 Casual Buyers",cm.index[2]:"🧾 Regular Spenders",cm.index[3]:"💎 Premium Buyers"}
        udf["Cluster_Label"] = udf["Cluster"].map(labels)
        X_pca = PCA(n_components=2).fit_transform(X)
        udf["PCA1"] = X_pca[:,0]; udf["PCA2"] = X_pca[:,1]
        return udf, inertias

    user_df, inertias = run_clustering(df)
    set_dark_style()

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### 📐 Elbow Method")
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(range(2,11),inertias,"o-",color="#bf5fff",linewidth=2,markersize=7,markerfacecolor="#ff2266",markeredgecolor="#bf5fff")
        ax.axvline(x=4,color="#00ff88",linestyle="--",linewidth=1.5,label="✅ Chosen k=4")
        ax.fill_between(range(2,11),inertias,alpha=0.15,color="#7b2fff")
        ax.set_title("📐 Elbow Curve"); ax.set_xlabel("Clusters (k)"); ax.set_ylabel("Inertia"); ax.legend()
        fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()
    with c2:
        st.markdown("### 🥧 Segment Distribution")
        counts = user_df["Cluster_Label"].value_counts()
        fig,ax = plt.subplots(figsize=(6,4))
        wedges,texts,autotexts = ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["#bf5fff","#00ff88","#00eeff","#ff8800"], startangle=140,
            wedgeprops={"edgecolor":"#0a0010","linewidth":2})
        for t in texts: t.set_color("#c0a0ff")
        for a in autotexts: a.set_color("#ffffff"); a.set_fontsize(9)
        ax.set_title("👥 Customer Segment Split")
        fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()

    st.markdown("### 🔭 Cluster Scatter Plot (PCA)")
    ccolors = {"💸 Budget Shoppers":"#00eeff","🛒 Casual Buyers":"#00ff88","🧾 Regular Spenders":"#ff8800","💎 Premium Buyers":"#bf5fff"}
    fig,ax = plt.subplots(figsize=(11,6))
    for label,grp in user_df.groupby("Cluster_Label"):
        ax.scatter(grp["PCA1"],grp["PCA2"],label=label,alpha=0.45,s=12,color=ccolors.get(label,"#fff"))
    ax.set_title("🔭 Customer Segments — PCA Space"); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    leg = ax.legend(facecolor="#0d0020",edgecolor="#7b2fff")
    for t in leg.get_texts(): t.set_color("#c0a0ff")
    fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()

    st.markdown("### 📋 Cluster Statistics")
    cs = user_df.groupby("Cluster_Label").agg(Avg_Purchase=("Avg_Purchase","mean"),Total_Purchase=("Total_Purchase","mean"),Num_Transactions=("Num_Transactions","mean"),Customer_Count=("User_ID","count")).round(2)
    st.dataframe(cs, width="stretch")

# ── TAB 3: ASSOCIATION RULES ──
with tab3:
    st.markdown("## 🔗 Association Rule Mining")
    st.markdown("*Finding hidden product purchase patterns using the Apriori algorithm*")
    c1,c2 = st.columns(2)
    with c1: min_support = st.slider("📏 Min Support", 0.05, 0.5, 0.10, 0.01)
    with c2: min_lift = st.slider("🚀 Min Lift", 1.0, 3.0, 1.2, 0.1)

    @st.cache_data
    def run_apriori(df, support, lift):
        basket = df.groupby("User_ID").apply(
            lambda x: list(set(["Cat1_"+str(int(c)) for c in x["Product_Category_1"] if c!=0]+
                               ["Cat2_"+str(int(c)) for c in x["Product_Category_2"] if c!=0]+
                               ["Cat3_"+str(int(c)) for c in x["Product_Category_3"] if c!=0]))
        ).tolist()
        te = TransactionEncoder()
        bdf = pd.DataFrame(te.fit_transform(basket), columns=te.columns_)
        freq = apriori(bdf, min_support=support, use_colnames=True)
        if len(freq)==0: return pd.DataFrame()
        rules = association_rules(freq, metric="lift", min_threshold=lift)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        return rules.sort_values("lift", ascending=False)

    rules_df = run_apriori(df, min_support, min_lift)
    set_dark_style()

    if rules_df.empty:
        st.warning("⚠️ No rules found — try lowering support or lift thresholds.")
    else:
        st.success(f"✅ Discovered **{len(rules_df)}** association rules!")
        st.markdown("### 🏹 Top Rules by Lift")
        top_rules = rules_df.head(15).copy()
        top_rules["rule"] = top_rules["antecedents"] + " ➜ " + top_rules["consequents"]
        fig,ax = plt.subplots(figsize=(11,6))
        colors_r = plt.cm.plasma(np.linspace(0.3,0.9,len(top_rules)))
        ax.barh(top_rules["rule"], top_rules["lift"], color=colors_r, edgecolor="#7b2fff", linewidth=0.6)
        ax.axvline(x=1, color="#ff2266", linestyle="--", linewidth=1.5, label="⚡ Lift=1 (random)")
        ax.set_title("🏹 Top 15 Association Rules by Lift"); ax.set_xlabel("Lift Score"); ax.legend()
        plt.tight_layout(); fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()
        st.markdown("### 📋 All Rules")
        st.dataframe(rules_df[["antecedents","consequents","support","confidence","lift"]].round(4), width="stretch")

# ── TAB 4: ANOMALY DETECTION ──
with tab4:
    st.markdown("## 🚨 Anomaly Detection")
    st.markdown("*Spotting unusual high-spenders using statistical outlier methods*")
    z_threshold = st.slider("🎚️ Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
    anomalies = df[df["Z_Score"] > z_threshold]
    c1,c2,c3 = st.columns(3)
    c1.metric("🚨 Anomalies Found", f"{len(anomalies):,}")
    c2.metric("💸 Max Purchase", f"${anomalies['Purchase'].max():,.0f}" if len(anomalies) else "N/A")
    c3.metric("📊 Anomaly Rate", f"{len(anomalies)/len(df)*100:.2f}%")
    set_dark_style()
    ca, cb = st.columns(2)
    with ca:
        st.markdown("### 🔴 Normal vs Anomalous")
        normal = df[~(df["Z_Score"] > z_threshold)]
        fig,ax = plt.subplots(figsize=(7,4))
        ax.scatter(range(min(5000,len(normal))), normal["Purchase"].head(5000), alpha=0.2, s=4, label="✅ Normal", color="#00eeff")
        ax.scatter(range(len(anomalies)), anomalies["Purchase"], alpha=0.95, s=30, label="🚨 Anomaly", color="#ff2266", edgecolors="#ff8800", linewidths=0.8)
        ax.set_title("🚨 Anomalies Detected"); ax.set_ylabel("Purchase (USD)")
        leg = ax.legend(facecolor="#0d0020",edgecolor="#7b2fff")
        for t in leg.get_texts(): t.set_color("#c0a0ff")
        fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()
    with cb:
        st.markdown("### 📈 Distribution + IQR Bound")
        Q1=df["Purchase"].quantile(0.25); Q3=df["Purchase"].quantile(0.75); IQR=Q3-Q1; upper=Q3+1.5*IQR
        fig,ax = plt.subplots(figsize=(7,4))
        n,bins,patches = ax.hist(df["Purchase"], bins=60, color="#7b2fff", alpha=0.75, edgecolor="#bf5fff", linewidth=0.3)
        ax.axvline(upper, color="#ff2266", linestyle="--", linewidth=2, label=f"🚧 IQR Upper: ${upper:,.0f}")
        ax.fill_betweenx([0,n.max()], upper, df["Purchase"].max(), alpha=0.1, color="#ff2266")
        ax.set_title("📈 Purchase Distribution"); ax.set_xlabel("Purchase (USD)"); ax.set_ylabel("Frequency")
        leg = ax.legend(facecolor="#0d0020",edgecolor="#7b2fff")
        for t in leg.get_texts(): t.set_color("#c0a0ff")
        fig.patch.set_facecolor("#0a0010"); st.pyplot(fig); plt.close()
    if len(anomalies)>0:
        st.markdown("### 🧑‍💼 Anomalous Spenders by Age Group")
        aa = anomalies.groupby("Age")["Purchase"].agg(["mean","count"]).reset_index()
        aa.columns = ["Age Group","Avg Anomalous Purchase","Count"]
        st.dataframe(aa, width="stretch")
        st.markdown("### 🔍 Sample Anomalous Transactions")
        st.dataframe(anomalies[["User_ID","Age","Gender","Occupation","City_Category","Purchase","Z_Score"]].head(20), width="stretch")

# ── TAB 5: INSIGHTS ──
with tab5:
    st.markdown("## 💡 Key Insights & Strategic Recommendations")
    st.markdown("*The story the data is telling InsightMart*")
    age_spend = df.groupby("Age")["Purchase"].mean().sort_values(ascending=False)
    gspend = df.groupby("Gender")["Purchase"].mean()
    top_cat = df["Product_Category_1"].value_counts().idxmax()
    city_spend = df.groupby("City_Category")["Purchase"].mean().sort_values(ascending=False)
    m_avg = gspend.get("M",0); f_avg = gspend.get("F",0)
    higher = "👨 Males" if m_avg>f_avg else "👩 Females"
    n_anom = int(df["Is_Anomaly"].sum())

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='neon-card' style='border-color:#00ff88;box-shadow:0 0 20px #00ff8833;'>
          <p style='font-family:Orbitron;color:#00ff88;font-size:0.9rem;text-shadow:0 0 8px #00ff88;'>🏆 TOP SPENDING AGE GROUP</p>
          <p style='font-size:1.4rem;color:#ffffff;'><b>{age_spend.index[0]}</b> years old</p>
          <p style='color:#90ffcc;'>Avg purchase: <b>${age_spend.iloc[0]:,.0f}</b></p>
        </div>
        <div class='neon-card' style='border-color:#00eeff;box-shadow:0 0 20px #00eeff33;'>
          <p style='font-family:Orbitron;color:#00eeff;font-size:0.9rem;text-shadow:0 0 8px #00eeff;'>⚧ GENDER SPENDING GAP</p>
          <p style='color:#ffffff;'>{higher} spend more on average</p>
          <p style='color:#90eeff;'>👨 Male: <b>${m_avg:,.0f}</b> &nbsp;|&nbsp; 👩 Female: <b>${f_avg:,.0f}</b></p>
        </div>
        <div class='neon-card'>
          <p style='font-family:Orbitron;color:#bf5fff;font-size:0.9rem;text-shadow:0 0 8px #bf5fff;'>🏙️ TOP CITY CATEGORY</p>
          <p style='color:#ffffff;'>City <b>{city_spend.index[0]}</b> dominates spending</p>
          <p style='color:#c0a0ff;'>Avg: <b>${city_spend.iloc[0]:,.0f}</b></p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        udf_i,_ = run_clustering(df)
        counts = udf_i["Cluster_Label"].value_counts()
        segs = "".join([f"<p style='color:#c0a0ff;margin:3px 0;'>{l}: <b>{c:,}</b> customers</p>" for l,c in counts.items()])
        st.markdown(f"""
        <div class='neon-card' style='border-color:#00ff88;box-shadow:0 0 20px #00ff8833;'>
          <p style='font-family:Orbitron;color:#00ff88;font-size:0.9rem;text-shadow:0 0 8px #00ff88;'>🛒 HOTTEST PRODUCT CATEGORY</p>
          <p style='font-size:1.4rem;color:#ffffff;'>Category <b>{top_cat}</b></p>
          <p style='color:#90ffcc;'>Most frequently purchased</p>
        </div>
        <div class='neon-card' style='border-color:#ff8800;box-shadow:0 0 20px #ff880033;'>
          <p style='font-family:Orbitron;color:#ff8800;font-size:0.9rem;text-shadow:0 0 8px #ff8800;'>🚨 ANOMALOUS SPENDERS</p>
          <p style='font-size:1.4rem;color:#ffffff;'><b>{n_anom:,}</b> flagged</p>
          <p style='color:#ffcc88;'>{n_anom/len(df)*100:.2f}% of all transactions</p>
        </div>
        <div class='neon-card'>
          <p style='font-family:Orbitron;color:#bf5fff;font-size:0.9rem;text-shadow:0 0 8px #bf5fff;'>👥 CUSTOMER SEGMENTS</p>
          {segs}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Strategic Recommendations")
    recs = [
        ("#00eeff","🎯","TARGET 26–35 AGE GROUP","Highest spending demographic — prioritise premium bundles and exclusive deals."),
        ("#bf5fff","👨","MALE-FOCUSED CAMPAIGNS","Male shoppers drive more revenue — tailor to electronics and gadgets."),
        ("#00ff88","🏙️","CITY A LOYALTY PROGRAMS","Highest spend capacity — launch VIP loyalty and early-access sales here."),
        ("#ff8800","🚨","MONITOR ANOMALOUS SPENDERS","High z-score transactions may indicate bulk buyers or fraud — flag for review."),
        ("#ff2266","🔗","CROSS-SELL WITH RULES","Bundle frequently co-purchased categories into combo deals to boost AOV."),
        ("#00eeff","💸","RE-ENGAGE BUDGET SHOPPERS","Targeted discounts to nudge Budget Shoppers up the spending ladder.")
    ]
    cr1,cr2 = st.columns(2)
    for i,(color,emoji,title,desc) in enumerate(recs):
        col = cr1 if i%2==0 else cr2
        col.markdown(f"""
        <div class='neon-card' style='border-color:{color}44;box-shadow:0 0 15px {color}22;margin-bottom:10px;'>
          <p style='margin:0 0 4px 0;font-family:Orbitron;color:{color};font-size:0.8rem;text-shadow:0 0 6px {color};'>{emoji} {title}</p>
          <p style='margin:0;color:#c0c0e0;font-size:0.82rem;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center;'><p style='font-family:Share Tech Mono;color:#5540aa;font-size:0.75rem;letter-spacing:2px;'>⚡ DATA MINING SUMMATIVE &nbsp;|&nbsp; 🛍️ INSIGHTMART ANALYTICS &nbsp;|&nbsp; 🎓 AI YEAR 1</p></div>", unsafe_allow_html=True)
