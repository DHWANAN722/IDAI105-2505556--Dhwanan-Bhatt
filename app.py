import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  CYBERPUNK NEO-NOIR THEME
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ BLACK FRIDAY // DATA CORE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CYBER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

/* ── Root variables ── */
:root {
    --neon-cyan:   #00ffe7;
    --neon-pink:   #ff2d78;
    --neon-yellow: #f9f002;
    --neon-purple: #bf00ff;
    --bg-void:     #010409;
    --bg-card:     #0d1117;
    --bg-panel:    #0a0f1a;
    --grid-line:   rgba(0,255,231,0.07);
    --text-main:   #e8f4f8;
    --text-dim:    #6a8a9a;
}

/* ── Global background ── */
.stApp {
    background-color: var(--bg-void) !important;
    background-image:
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px),
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,255,231,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 100% 80%, rgba(255,45,120,0.06) 0%, transparent 60%);
    background-size: 40px 40px, 40px 40px, 100% 100%, 100% 100%;
    font-family: 'Rajdhani', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040810 0%, #080d1a 100%) !important;
    border-right: 1px solid rgba(0,255,231,0.2) !important;
    box-shadow: 4px 0 30px rgba(0,255,231,0.05);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── All text ── */
html, body, [class*="css"], p, div, span, label {
    color: var(--text-main) !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* ── Headers ── */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.08em;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,255,231,0.25) !important;
    border-radius: 4px !important;
    padding: 16px !important;
    position: relative;
    box-shadow: 0 0 20px rgba(0,255,231,0.08), inset 0 0 20px rgba(0,0,0,0.5) !important;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
}
[data-testid="stMetricLabel"] { color: var(--neon-cyan) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.15em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--neon-yellow) !important; font-family: 'Orbitron', sans-serif !important; font-size: 1.6rem !important; font-weight: 700; text-shadow: 0 0 20px rgba(249,240,2,0.5); }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-bottom: 1px solid rgba(0,255,231,0.2) !important;
    gap: 4px;
}
[data-baseweb="tab"] {
    color: var(--text-dim) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em;
    border-bottom: 2px solid transparent !important;
    padding: 12px 20px !important;
    transition: all 0.3s ease;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--neon-cyan) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
    background: rgba(0,255,231,0.05) !important;
    text-shadow: 0 0 12px rgba(0,255,231,0.8);
}

/* ── Buttons ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-card) !important;
    border: 1px dashed rgba(0,255,231,0.4) !important;
    border-radius: 4px !important;
}

/* ── Sliders ── */
[data-baseweb="slider"] [role="slider"] {
    background: var(--neon-cyan) !important;
    box-shadow: 0 0 10px var(--neon-cyan) !important;
}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: var(--neon-cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,255,231,0.2) !important;
    border-radius: 4px !important;
}

/* ── Success / Info / Warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stSuccess { background: rgba(0,255,100,0.08) !important; border-left: 3px solid #00ff64 !important; }
.stInfo    { background: rgba(0,255,231,0.08) !important; border-left: 3px solid var(--neon-cyan) !important; }
.stWarning { background: rgba(255,45,120,0.08) !important; border-left: 3px solid var(--neon-pink) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--neon-cyan); border-radius: 2px; }

/* ── Divider ── */
hr { border-color: rgba(0,255,231,0.15) !important; }

/* ── Sidebar upload label ── */
[data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--text-dim) !important; }
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)

# ── Matplotlib dark cyberpunk theme ──────────────────────────────────────────
NEON_CYAN   = "#00ffe7"
NEON_PINK   = "#ff2d78"
NEON_YELLOW = "#f9f002"
NEON_PURPLE = "#bf00ff"
NEON_GREEN  = "#00ff64"
NEON_ORANGE = "#ff6b00"
BG_VOID     = "#010409"
BG_CARD     = "#0d1117"

CYBER_PALETTE = [NEON_CYAN, NEON_PINK, NEON_YELLOW, NEON_PURPLE, NEON_GREEN, NEON_ORANGE]

def cyber_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor("#080d16")
    ax.tick_params(colors="#6a8a9a", labelsize=8)
    ax.xaxis.label.set_color("#6a8a9a")
    ax.yaxis.label.set_color("#6a8a9a")
    ax.title.set_color(NEON_CYAN)
    for spine in ax.spines.values():
        spine.set_edgecolor("rgba(0,255,231,0.15)")
        spine.set_linewidth(0.5)
    ax.grid(True, color="rgba(0,255,231,0.06)", linewidth=0.5, linestyle="--")
    return fig, ax

def cyber_bar(labels, values, color=NEON_CYAN, title="", xlabel="", ylabel="", h_bar=False):
    fig, ax = cyber_fig(8, 4)
    bar_colors = [color] * len(values)
    if h_bar:
        bars = ax.barh(labels, values, color=bar_colors, edgecolor="none")
        for bar, val in zip(bars, values):
            bar.set_alpha(0.85)
            # neon glow effect via twin bar
            ax.barh([bar.get_y() + bar.get_height()/2], [val],
                    height=bar.get_height()*0.3,
                    color=color, alpha=0.3, edgecolor="none",
                    left=0)
    else:
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="none")
        for bar in bars:
            bar.set_alpha(0.85)
    ax.set_title(title, fontfamily="monospace", fontsize=10, pad=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.xticks(rotation=30 if not h_bar else 0, fontsize=8)
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 10px 0 20px 0;">
  <div style="font-family:'Orbitron',sans-serif; font-size:2.2rem; font-weight:900;
              color:#00ffe7; text-shadow: 0 0 30px #00ffe7, 0 0 60px rgba(0,255,231,0.4);
              letter-spacing:0.15em;">
    ⚡ BLACK FRIDAY // DATA CORE ⚡
  </div>
  <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#ff2d78;
              letter-spacing:0.3em; margin-top:6px; text-shadow: 0 0 10px #ff2d78;">
    [ INSIGHTMART ANALYTICS // NEURAL MINING SYSTEM // v2.0 ]
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr style="border:none; height:1px; background: linear-gradient(90deg, transparent, #00ffe7, #ff2d78, transparent);">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="font-family:'Orbitron',sans-serif; font-size:0.9rem; color:#00ffe7;
            text-shadow:0 0 10px #00ffe7; letter-spacing:0.1em; margin-bottom:16px;">
  ◈ DATA UPLINK
</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("▸ INJECT DATASET [train.csv]", type="csv")

st.sidebar.markdown("""
<div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#6a8a9a;
            margin-top:20px; line-height:1.8;">
  ◆ SYSTEM STATUS<br>
  ├─ ENGINE: SKLEARN v1.8<br>
  ├─ ALGO: K-MEANS + APRIORI<br>
  ├─ ANOMALY: Z-SCORE / IQR<br>
  └─ VIZ: MATPLOTLIB CORE
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;
                border:1px solid rgba(0,255,231,0.15); border-radius:8px;
                background:rgba(0,255,231,0.02); margin-top:30px;">
      <div style="font-size:3rem;">💾</div>
      <div style="font-family:'Orbitron',sans-serif; font-size:1rem; color:#00ffe7;
                  letter-spacing:0.1em; margin:16px 0 8px;">AWAITING DATA INJECTION</div>
      <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#6a8a9a;">
        Upload train.csv via the sidebar uplink to initialize neural scan
      </div>
      <div style="margin-top:20px;">
        <a href="https://drive.google.com/drive/folders/13DxtCVj3S_AAYXG5THw2mmr6_VA1N3L9"
           style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#ff2d78;
                  text-decoration:none; border:1px solid #ff2d78; padding:8px 16px;
                  border-radius:2px; letter-spacing:0.1em;">
          ⬇ DOWNLOAD DATASET
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)
    df["Product_Category_2"] = df["Product_Category_2"].fillna(0).astype(int)
    df["Product_Category_3"] = df["Product_Category_3"].fillna(0).astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Gender_Encoded"] = df["Gender"].map({"M": 0, "F": 1})
    age_map = {"0-17":1,"18-25":2,"26-35":3,"36-45":4,"46-50":5,"51-55":6,"55+":7}
    df["Age_Encoded"] = df["Age"].map(age_map)
    df["City_Encoded"] = df["City_Category"].map({"A":1,"B":2,"C":3})
    df["Stay_Encoded"] = df["Stay_In_Current_City_Years"].replace("4+",4).astype(int)
    df["Purchase_Normalized"] = StandardScaler().fit_transform(df[["Purchase"]])
    df["Z_Score"] = np.abs(stats.zscore(df["Purchase"]))
    df["Is_Anomaly"] = df["Z_Score"] > 3
    return df

try:
    df = load_and_clean(uploaded)
except Exception as e:
    st.error(f"⚠ DATA CORRUPTION DETECTED: {e}")
    st.stop()

st.sidebar.markdown(f"""
<div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
            color:#00ff64; margin-top:16px; line-height:1.8;
            border:1px solid rgba(0,255,100,0.2); padding:10px; border-radius:4px;
            background:rgba(0,255,100,0.03);">
  ✔ LINK ESTABLISHED<br>
  ├─ RECORDS: {len(df):,}<br>
  ├─ USERS: {df['User_ID'].nunique():,}<br>
  └─ PRODUCTS: {df['Product_ID'].nunique():,}
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  CLUSTERING
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def run_clustering(_df):
    user_df = _df.groupby("User_ID").agg(
        Age_Encoded=("Age_Encoded","first"), Gender_Encoded=("Gender_Encoded","first"),
        Occupation=("Occupation","first"), Marital_Status=("Marital_Status","first"),
        Avg_Purchase=("Purchase","mean"), Total_Purchase=("Purchase","sum"),
        Num_Transactions=("Purchase","count")
    ).reset_index()
    features = ["Age_Encoded","Gender_Encoded","Occupation","Marital_Status","Avg_Purchase","Num_Transactions"]
    X = StandardScaler().fit_transform(user_df[features])
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in range(2,11)]
    user_df["Cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    cm = user_df.groupby("Cluster")["Avg_Purchase"].mean().sort_values()
    lmap = {cm.index[0]:"🟦 BUDGET UNIT",cm.index[1]:"🟩 CASUAL NODE",
            cm.index[2]:"🟧 REGULAR GRID",cm.index[3]:"🟥 PREMIUM CORE"}
    user_df["Cluster_Label"] = user_df["Cluster"].map(lmap)
    Xp = PCA(n_components=2).fit_transform(X)
    user_df["PCA1"] = Xp[:,0]; user_df["PCA2"] = Xp[:,1]
    return user_df, inertias

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡  EDA SCAN", "🧬  CLUSTER MAP", "🔮  RULE ENGINE",
    "☢️  ANOMALY TRACE", "💀  INTEL REPORT"
])

# ════════ TAB 1 – EDA ════════
with tab1:
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:1.1rem;color:#00ffe7;letter-spacing:0.12em;text-shadow:0 0 12px #00ffe7;margin-bottom:20px;">◈ EXPLORATORY DATA SCAN</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("⚡ TRANSACTIONS", f"{len(df):,}")
    c2.metric("👤 UNIQUE USERS", f"{df['User_ID'].nunique():,}")
    c3.metric("📦 PRODUCTS", f"{df['Product_ID'].nunique():,}")
    c4.metric("💰 AVG SPEND", f"${df['Purchase'].mean():,.0f}")

    st.markdown('<hr style="border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(0,255,231,0.3),transparent);margin:20px 0;">', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin-bottom:8px;">▸ AVG SPEND BY AGE COHORT</div>', unsafe_allow_html=True)
        age_order = ["0-17","18-25","26-35","36-45","46-50","51-55","55+"]
        age_data = df.groupby("Age")["Purchase"].mean().reindex(age_order).dropna()
        fig, ax = cyber_fig(7, 4)
        bars = ax.bar(age_data.index, age_data.values, color=NEON_CYAN, edgecolor="none", alpha=0.85)
        # glow
        ax.bar(age_data.index, age_data.values, color=NEON_CYAN, edgecolor="none", alpha=0.15, width=0.8)
        ax.set_ylabel("Avg Purchase (USD)", color="#6a8a9a", fontsize=8)
        ax.set_title("AGE COHORT // SPEND MATRIX", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
        plt.xticks(rotation=30, fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cb:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin-bottom:8px;">▸ GENDER SPEND DIFFERENTIAL</div>', unsafe_allow_html=True)
        gd = df.groupby("Gender")["Purchase"].mean()
        fig, ax = cyber_fig(5, 4)
        bars = ax.bar(["♂ MALE","♀ FEMALE"], [gd.get("M",0),gd.get("F",0)],
                      color=[NEON_CYAN, NEON_PINK], edgecolor="none", alpha=0.85)
        ax.bar(["♂ MALE","♀ FEMALE"], [gd.get("M",0),gd.get("F",0)],
               color=[NEON_CYAN, NEON_PINK], edgecolor="none", alpha=0.2)
        ax.set_ylabel("Avg Purchase (USD)", color="#6a8a9a", fontsize=8)
        ax.set_title("GENDER // SPEND SPLIT", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    cc, cd = st.columns(2)
    with cc:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin-bottom:8px;">▸ TOP PRODUCT CATEGORY FREQUENCY</div>', unsafe_allow_html=True)
        tc = df["Product_Category_1"].value_counts().head(10)
        fig, ax = cyber_fig(7, 4)
        ax.bar(tc.index.astype(str), tc.values, color=NEON_YELLOW, edgecolor="none", alpha=0.85)
        ax.bar(tc.index.astype(str), tc.values, color=NEON_YELLOW, edgecolor="none", alpha=0.15)
        ax.set_xlabel("Category ID", color="#6a8a9a", fontsize=8)
        ax.set_ylabel("Transaction Count", color="#6a8a9a", fontsize=8)
        ax.set_title("PRODUCT CATEGORY // FREQUENCY SCAN", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cd:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin-bottom:8px;">▸ CATEGORY REVENUE MAP</div>', unsafe_allow_html=True)
        ac = df.groupby("Product_Category_1")["Purchase"].mean().sort_values(ascending=True).tail(10)
        fig, ax = cyber_fig(7, 4)
        ax.barh(ac.index.astype(str), ac.values, color=NEON_PURPLE, edgecolor="none", alpha=0.85)
        ax.barh(ac.index.astype(str), ac.values, color=NEON_PURPLE, edgecolor="none", alpha=0.15)
        ax.set_xlabel("Avg Purchase (USD)", color="#6a8a9a", fontsize=8)
        ax.set_title("CATEGORY // AVG REVENUE SIGNAL", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin:16px 0 8px;">▸ FEATURE CORRELATION MATRIX</div>', unsafe_allow_html=True)
    num_cols = ["Age_Encoded","Gender_Encoded","Occupation","City_Encoded","Stay_Encoded",
                "Marital_Status","Product_Category_1","Product_Category_2","Product_Category_3","Purchase"]
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor("#080d16")
    cmap = sns.diverging_palette(180, 330, s=90, l=40, as_cmap=True)
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap=cmap,
                ax=ax, linewidths=0.5, linecolor="#0d1117",
                annot_kws={"size":7, "color":"white"},
                cbar_kws={"shrink":0.8})
    ax.set_title("NEURAL CORRELATION MATRIX", color=NEON_CYAN, fontfamily="monospace", fontsize=10, pad=15)
    ax.tick_params(colors="#6a8a9a", labelsize=7)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════ TAB 2 – CLUSTERING ════════
with tab2:
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:1.1rem;color:#00ffe7;letter-spacing:0.12em;text-shadow:0 0 12px #00ffe7;margin-bottom:20px;">◈ CUSTOMER CLUSTER MAP</div>', unsafe_allow_html=True)
    try:
        user_df, inertias = run_clustering(df)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin-bottom:8px;">▸ ELBOW CALIBRATION SIGNAL</div>', unsafe_allow_html=True)
            fig, ax = cyber_fig(6,4)
            ax.plot(range(2,11), inertias, color=NEON_CYAN, linewidth=2, marker="o",
                    markersize=6, markerfacecolor=NEON_PINK, markeredgecolor=NEON_PINK)
            ax.fill_between(range(2,11), inertias, alpha=0.1, color=NEON_CYAN)
            ax.axvline(x=4, color=NEON_PINK, linestyle="--", linewidth=1.5, alpha=0.8, label="OPTIMAL k=4")
            ax.set_xlabel("Cluster Count (k)", color="#6a8a9a", fontsize=8)
            ax.set_ylabel("Inertia", color="#6a8a9a", fontsize=8)
            ax.set_title("ELBOW CALIBRATION // k-SCAN", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
            ax.legend(facecolor=BG_CARD, edgecolor=NEON_PINK, labelcolor=NEON_PINK, fontsize=7)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin-bottom:8px;">▸ SEGMENT DISTRIBUTION</div>', unsafe_allow_html=True)
            counts = user_df["Cluster_Label"].value_counts()
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor(BG_CARD)
            ax.set_facecolor(BG_CARD)
            wedge_colors = [NEON_CYAN, NEON_GREEN, NEON_ORANGE, NEON_PINK]
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=wedge_colors, startangle=90,
                wedgeprops={"edgecolor": BG_VOID, "linewidth": 2},
                textprops={"color":"#e8f4f8","fontsize":7,"fontfamily":"monospace"}
            )
            for at in autotexts:
                at.set_color(BG_VOID); at.set_fontsize(7); at.set_fontweight("bold")
            ax.set_title("USER SEGMENT // DISTRIBUTION", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin:16px 0 8px;">▸ PCA CLUSTER SCATTER // 2D PROJECTION</div>', unsafe_allow_html=True)
        CMAP = {"🟦 BUDGET UNIT":NEON_CYAN,"🟩 CASUAL NODE":NEON_GREEN,
                "🟧 REGULAR GRID":NEON_ORANGE,"🟥 PREMIUM CORE":NEON_PINK}
        fig, ax = cyber_fig(12, 5)
        for lbl, grp in user_df.groupby("Cluster_Label"):
            c = CMAP.get(lbl, "#ffffff")
            ax.scatter(grp["PCA1"], grp["PCA2"], label=lbl, alpha=0.6, s=12,
                       color=c, edgecolors="none")
            ax.scatter(grp["PCA1"], grp["PCA2"], alpha=0.08, s=30, color=c, edgecolors="none")
        ax.set_xlabel("PCA DIMENSION 1", color="#6a8a9a", fontsize=8)
        ax.set_ylabel("PCA DIMENSION 2", color="#6a8a9a", fontsize=8)
        ax.set_title("CUSTOMER CLUSTER MAP // PCA SPACE", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
        ax.legend(facecolor=BG_CARD, edgecolor="rgba(0,255,231,0.2)", labelcolor="#e8f4f8", fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin:16px 0 8px;">▸ CLUSTER STATS TABLE</div>', unsafe_allow_html=True)
        st.dataframe(user_df.groupby("Cluster_Label").agg(
            Avg_Purchase=("Avg_Purchase","mean"), Total_Purchase=("Total_Purchase","mean"),
            Num_Transactions=("Num_Transactions","mean"), User_Count=("User_ID","count")
        ).round(2))
    except Exception as e:
        st.error(f"⚠ CLUSTER ENGINE FAILURE: {e}")

# ════════ TAB 3 – ASSOCIATION RULES ════════
with tab3:
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:1.1rem;color:#00ffe7;letter-spacing:0.12em;text-shadow:0 0 12px #00ffe7;margin-bottom:20px;">◈ ASSOCIATION RULE ENGINE</div>', unsafe_allow_html=True)

    col_s, col_l = st.columns(2)
    with col_s:
        min_support = st.slider("⚙ MIN SUPPORT THRESHOLD", 0.05, 0.5, 0.10, 0.01)
    with col_l:
        min_lift = st.slider("⚙ MIN LIFT THRESHOLD", 1.0, 3.0, 1.2, 0.1)

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
            st.warning("⚠ NO SIGNAL DETECTED — Lower support/lift to expand scan radius")
        else:
            st.success(f"✔ {len(rdf)} ASSOCIATION RULES DECODED")
            top = rdf.head(15).copy()
            top["rule"] = top["antecedents"]+" ⟶ "+top["consequents"]
            fig, ax = cyber_fig(11, 6)
            bars = ax.barh(top["rule"], top["lift"], color=NEON_ORANGE, edgecolor="none", alpha=0.85)
            ax.barh(top["rule"], top["lift"], color=NEON_ORANGE, edgecolor="none", alpha=0.12)
            ax.axvline(x=1, color=NEON_PINK, linestyle="--", linewidth=1, alpha=0.7, label="LIFT = 1 [BASELINE]")
            ax.set_xlabel("LIFT SCORE", color="#6a8a9a", fontsize=8)
            ax.set_title("TOP 15 RULES // LIFT SIGNAL STRENGTH", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
            ax.legend(facecolor=BG_CARD, edgecolor=NEON_PINK, labelcolor=NEON_PINK, fontsize=7)
            ax.tick_params(labelsize=7)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin:16px 0 8px;">▸ FULL RULE MATRIX</div>', unsafe_allow_html=True)
            st.dataframe(rdf[["antecedents","consequents","support","confidence","lift"]].round(4))
    except Exception as e:
        st.error(f"⚠ RULE ENGINE FAULT: {e}")

# ════════ TAB 4 – ANOMALY ════════
with tab4:
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:1.1rem;color:#ff2d78;letter-spacing:0.12em;text-shadow:0 0 12px #ff2d78;margin-bottom:20px;">◈ ANOMALY TRACE // OUTLIER DETECTION</div>', unsafe_allow_html=True)

    z_thresh = st.slider("☢ Z-SCORE THRESHOLD", 2.0, 4.0, 3.0, 0.1)
    anom = df[df["Z_Score"] > z_thresh]
    c1,c2,c3 = st.columns(3)
    c1.metric("☢ ANOMALIES FLAGGED", f"{len(anom):,}")
    c2.metric("💀 MAX SPEND DETECTED", f"${anom['Purchase'].max():,.0f}" if len(anom) else "NONE")
    c3.metric("📊 ANOMALY RATE", f"{len(anom)/len(df)*100:.2f}%")

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin-bottom:8px;">▸ TRANSACTION ANOMALY SCATTER</div>', unsafe_allow_html=True)
        norm = df[df["Z_Score"] <= z_thresh]
        fig, ax = cyber_fig(7, 4)
        ax.scatter(range(min(5000,len(norm))), norm["Purchase"].iloc[:5000],
                   alpha=0.3, s=4, label="NORMAL", color=NEON_CYAN, edgecolors="none")
        if len(anom):
            ax.scatter(range(len(anom)), anom["Purchase"],
                       alpha=1.0, s=30, label="⚠ ANOMALY", color=NEON_PINK, edgecolors=NEON_PINK,
                       linewidths=0.5, zorder=5)
            ax.scatter(range(len(anom)), anom["Purchase"],
                       alpha=0.3, s=80, color=NEON_PINK, edgecolors="none", zorder=4)
        ax.set_ylabel("Purchase (USD)", color="#6a8a9a", fontsize=8)
        ax.set_title("PURCHASE ANOMALY SCATTER", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
        ax.legend(facecolor=BG_CARD, edgecolor=NEON_PINK, labelcolor="#e8f4f8", fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cb:
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin-bottom:8px;">▸ PURCHASE DISTRIBUTION + IQR BOUND</div>', unsafe_allow_html=True)
        Q1,Q3 = df["Purchase"].quantile(0.25), df["Purchase"].quantile(0.75)
        upper = Q3+1.5*(Q3-Q1)
        fig, ax = cyber_fig(7, 4)
        ax.hist(df["Purchase"], bins=60, color=NEON_CYAN, alpha=0.6, edgecolor="none")
        ax.hist(df["Purchase"], bins=60, color=NEON_CYAN, alpha=0.1, edgecolor="none")
        ax.axvline(upper, color=NEON_PINK, linestyle="--", linewidth=1.5,
                   label=f"IQR UPPER BOUND: ${upper:,.0f}")
        ax.set_xlabel("Purchase (USD)", color="#6a8a9a", fontsize=8)
        ax.set_ylabel("Frequency", color="#6a8a9a", fontsize=8)
        ax.set_title("SPEND DISTRIBUTION // IQR BOUNDARY", color=NEON_CYAN, fontfamily="monospace", fontsize=9)
        ax.legend(facecolor=BG_CARD, edgecolor=NEON_PINK, labelcolor=NEON_PINK, fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    if len(anom):
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin:16px 0 8px;">▸ ANOMALY BREAKDOWN BY AGE COHORT</div>', unsafe_allow_html=True)
        st.dataframe(anom.groupby("Age")["Purchase"].agg(["mean","count"]).reset_index().rename(columns={"mean":"Avg Spend","count":"Count"}))
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#ff2d78;letter-spacing:0.1em;margin:16px 0 8px;">▸ FLAGGED TRANSACTIONS</div>', unsafe_allow_html=True)
        st.dataframe(anom[["User_ID","Age","Gender","Occupation","City_Category","Purchase","Z_Score"]].head(20))

# ════════ TAB 5 – INSIGHTS ════════
with tab5:
    st.markdown('<div style="font-family:Orbitron,sans-serif;font-size:1.1rem;color:#f9f002;letter-spacing:0.12em;text-shadow:0 0 12px #f9f002;margin-bottom:20px;">◈ INTEL REPORT // MISSION DEBRIEF</div>', unsafe_allow_html=True)
    try:
        age_spend = df.groupby("Age")["Purchase"].mean().sort_values(ascending=False)
        gender_spend = df.groupby("Gender")["Purchase"].mean()
        top_cat = df["Product_Category_1"].value_counts().idxmax()
        city_spend = df.groupby("City_Category")["Purchase"].mean().sort_values(ascending=False)
        udf, _ = run_clustering(df)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style="background:rgba(0,255,231,0.04);border:1px solid rgba(0,255,231,0.2);
                        border-radius:4px;padding:16px;margin-bottom:12px;">
              <div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#00ffe7;letter-spacing:0.15em;">🏆 TOP SPENDING COHORT</div>
              <div style="font-family:Orbitron,sans-serif;font-size:1.4rem;color:#f9f002;
                          text-shadow:0 0 15px rgba(249,240,2,0.5);margin-top:6px;">AGE: {age_spend.index[0]}</div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#e8f4f8;margin-top:4px;">AVG SPEND: ${age_spend.iloc[0]:,.0f}</div>
            </div>
            <div style="background:rgba(255,45,120,0.04);border:1px solid rgba(255,45,120,0.2);
                        border-radius:4px;padding:16px;margin-bottom:12px;">
              <div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#ff2d78;letter-spacing:0.15em;">👔 GENDER SPEND SPLIT</div>
              <div style="font-family:Orbitron,sans-serif;font-size:0.9rem;color:#e8f4f8;margin-top:6px;">
                ♂ MALE: <span style="color:#00ffe7;">${gender_spend.get('M',0):,.0f}</span> &nbsp;|&nbsp;
                ♀ FEMALE: <span style="color:#ff2d78;">${gender_spend.get('F',0):,.0f}</span>
              </div>
            </div>
            <div style="background:rgba(191,0,255,0.04);border:1px solid rgba(191,0,255,0.2);
                        border-radius:4px;padding:16px;">
              <div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#bf00ff;letter-spacing:0.15em;">🏙️ TOP CITY NODE</div>
              <div style="font-family:Orbitron,sans-serif;font-size:1.4rem;color:#f9f002;
                          text-shadow:0 0 15px rgba(249,240,2,0.5);margin-top:6px;">CITY: {city_spend.index[0]}</div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#e8f4f8;margin-top:4px;">AVG SPEND: ${city_spend.iloc[0]:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div style="background:rgba(249,240,2,0.04);border:1px solid rgba(249,240,2,0.2);
                        border-radius:4px;padding:16px;margin-bottom:12px;">
              <div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#f9f002;letter-spacing:0.15em;">🛒 TOP PRODUCT CATEGORY</div>
              <div style="font-family:Orbitron,sans-serif;font-size:1.4rem;color:#00ffe7;
                          text-shadow:0 0 15px rgba(0,255,231,0.5);margin-top:6px;">CAT #{top_cat}</div>
            </div>
            <div style="background:rgba(255,45,120,0.04);border:1px solid rgba(255,45,120,0.2);
                        border-radius:4px;padding:16px;margin-bottom:12px;">
              <div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#ff2d78;letter-spacing:0.15em;">☢️ ANOMALY STATUS</div>
              <div style="font-family:Orbitron,sans-serif;font-size:1.4rem;color:#ff2d78;
                          text-shadow:0 0 15px rgba(255,45,120,0.5);margin-top:6px;">{df['Is_Anomaly'].sum():,} FLAGGED</div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#e8f4f8;margin-top:4px;">{df['Is_Anomaly'].sum()/len(df)*100:.2f}% OF ALL TRANSACTIONS</div>
            </div>
            <div style="background:rgba(0,255,100,0.04);border:1px solid rgba(0,255,100,0.2);
                        border-radius:4px;padding:16px;">
              <div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#00ff64;letter-spacing:0.15em;">👥 SEGMENT INTEL</div>
              {"".join([f'<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#e8f4f8;margin-top:6px;">{lbl}: <span style=\'color:#f9f002;\'>{cnt:,}</span></div>' for lbl, cnt in udf["Cluster_Label"].value_counts().items()])}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr style="border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(249,240,2,0.4),transparent);margin:24px 0;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:Orbitron,sans-serif;font-size:0.85rem;color:#f9f002;
                    letter-spacing:0.1em;text-shadow:0 0 10px rgba(249,240,2,0.5);margin-bottom:16px;">
          ◈ STRATEGIC RECOMMENDATIONS
        </div>
        <div style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#e8f4f8;
                    line-height:2.2; background:rgba(0,0,0,0.3); padding:20px;
                    border:1px solid rgba(249,240,2,0.15); border-radius:4px;">
          <span style="color:#00ffe7;">01 //</span> TARGET <span style="color:#f9f002;">26–35 AGE COHORT</span> — highest avg spend, maximize premium bundle campaigns<br>
          <span style="color:#00ffe7;">02 //</span> <span style="color:#f9f002;">MALE SHOPPERS</span> dominate revenue — build male-centric category promotions<br>
          <span style="color:#00ffe7;">03 //</span> DEPLOY loyalty programs in <span style="color:#f9f002;">CITY A</span> — highest spend capacity node<br>
          <span style="color:#00ffe7;">04 //</span> FLAG <span style="color:#ff2d78;">ANOMALOUS TRANSACTIONS</span> — investigate bulk buyers and potential fraud vectors<br>
          <span style="color:#00ffe7;">05 //</span> EXECUTE <span style="color:#f9f002;">CROSS-SELL COMBOS</span> from top association rules — bundle co-purchased categories<br>
          <span style="color:#00ffe7;">06 //</span> ACTIVATE <span style="color:#bf00ff;">BUDGET SHOPPERS</span> with discount triggers — move them up the spend ladder
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠ INTEL CORE FAULT: {e}")

st.markdown('<hr style="border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(0,255,231,0.2),rgba(255,45,120,0.2),transparent);margin-top:30px;">', unsafe_allow_html=True)
st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#2a4a5a;text-align:center;padding:8px;">[ DATA MINING SUMMATIVE // INSIGHTMART ANALYTICS // BLACK FRIDAY NEURAL CORE // v2.0 ]</div>', unsafe_allow_html=True)
