"""
Beyond Discounts: Data-Driven Black Friday Sales Insights
Streamlit Dashboard – Dark Theme Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Black Friday Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0a0f !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #12101e 100%) !important;
    border-right: 1px solid #2a2640 !important;
}
[data-testid="stSidebar"] * { color: #c8c4e0 !important; font-family: 'DM Sans', sans-serif !important; }

[data-testid="stMainBlockContainer"] {
    animation: pageSlideIn 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
}
@keyframes pageSlideIn {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
}

h1, h2, h3, h4, h5, h6, p, span, div, label { color: #e8e6f0 !important; font-family: 'DM Sans', sans-serif !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #16132a 0%, #1e1a33 100%) !important;
    border: 1px solid #2e2a50 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    transition: transform 0.25s ease, border-color 0.25s ease !important;
    animation: cardPop 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px) !important; border-color: #ff6b35 !important; }
[data-testid="stMetricValue"] { color: #ff6b35 !important; font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #9a96b8 !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
@keyframes cardPop { from { opacity: 0; transform: scale(0.94); } to { opacity: 1; transform: scale(1); } }

[data-testid="stTabs"] button { background: transparent !important; color: #7a76a0 !important; border: none !important; font-family: 'DM Sans', sans-serif !important; transition: color 0.2s ease !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #ff6b35 !important; border-bottom: 2px solid #ff6b35 !important; }
[data-testid="stTabs"] button:hover { color: #ffb38a !important; }

[data-testid="stFileUploader"] { background: #16132a !important; border: 1px dashed #3a3660 !important; border-radius: 10px !important; }
[data-testid="stDataFrame"] { background: #0f0d1e !important; border-radius: 10px !important; overflow: hidden !important; }
iframe { filter: invert(0.88) hue-rotate(180deg) !important; border-radius: 10px !important; }
[data-testid="stRadio"] label { padding: 8px 12px !important; border-radius: 8px !important; transition: background 0.2s ease !important; cursor: pointer !important; }
[data-testid="stRadio"] label:hover { background: #1e1a33 !important; }
hr { border-color: #2a2640 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2e2a50; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #ff6b35; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 50%, #ff6b35 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    line-height: 1.1; margin-bottom: 0.3rem;
}
@keyframes shimmer { from { background-position: 200% center; } to { background-position: -200% center; } }

.hero-sub { font-family: 'DM Sans', sans-serif; font-size: 1rem; color: #7a76a0 !important; letter-spacing: 0.04em; margin-bottom: 1.5rem; }

.section-header {
    font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700;
    color: #e8e6f0 !important;
    border-left: 4px solid #ff6b35; padding-left: 14px;
    margin: 1.5rem 0 1rem 0;
    animation: slideRight 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
}
@keyframes slideRight { from { opacity: 0; transform: translateX(-16px); } to { opacity: 1; transform: translateX(0); } }

.insight-card {
    background: linear-gradient(135deg, #13102a 0%, #1a1630 100%);
    border: 1px solid #2a2640; border-left: 4px solid #ff6b35;
    border-radius: 12px; padding: 16px 20px; margin: 10px 0;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    animation: fadeUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.insight-card:hover { transform: translateX(6px); border-color: #ff9a6b; box-shadow: 0 4px 24px rgba(255,107,53,0.15); }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
.insight-card p { color: #b8b4d0 !important; margin: 0; font-size: 0.95rem; }

.stat-pill { display: inline-block; background: #1e1a35; border: 1px solid #3a3660; border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; color: #c8c4e0 !important; margin: 3px; }
.sidebar-brand { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800; background: linear-gradient(135deg, #ff6b35, #f7c59f); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.nav-hint { font-size: 0.72rem; color: #4a4668 !important; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 1rem; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0d1e', 'axes.facecolor': '#0f0d1e',
    'axes.edgecolor': '#2a2640', 'axes.labelcolor': '#c8c4e0',
    'axes.titlecolor': '#e8e6f0', 'xtick.color': '#7a76a0',
    'ytick.color': '#7a76a0', 'text.color': '#e8e6f0',
    'grid.color': '#1e1a33', 'grid.alpha': 0.6,
    'legend.facecolor': '#16132a', 'legend.edgecolor': '#2a2640',
    'figure.dpi': 130,
})

ACCENT  = '#ff6b35'
PALETTE = ['#ff6b35', '#7c6af7', '#3dcfcf', '#f7c559', '#e06bf5', '#5dde7a']
CLUSTER_COLORS = {
    'Budget Shoppers':  '#f7c559',
    'Casual Buyers':    '#7c6af7',
    'Regular Spenders': '#3dcfcf',
    'Premium Buyers':   '#ff6b35'
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ InsightMart</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#4a4668;font-size:0.78rem;margin-bottom:1rem;">BLACK FRIDAY ANALYTICS</div>', unsafe_allow_html=True)
    st.divider()
    uploaded = st.file_uploader("Upload BlackFriday.csv", type=["csv"], label_visibility="collapsed")
    st.markdown('<div class="nav-hint">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Overview",
        "🔍  EDA & Visualizations",
        "🗂️  Customer Clustering",
        "🔗  Association Rules",
        "🚨  Anomaly Detection",
        "💡  Key Insights"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown('<div style="color:#4a4668;font-size:0.72rem;text-align:center;">Data Mining · Year 1 · Summative</div>', unsafe_allow_html=True)

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_prep(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        np.random.seed(42)
        n = 5000
        ages = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']
        df = pd.DataFrame({
            'User_ID': np.random.randint(1000000, 1006000, n),
            'Product_ID': [f'P{np.random.randint(10000,99999)}' for _ in range(n)],
            'Gender': np.random.choice(['M','F'], n, p=[0.75,0.25]),
            'Age': np.random.choice(ages, n),
            'Occupation': np.random.randint(0, 21, n),
            'City_Category': np.random.choice(['A','B','C'], n),
            'Stay_In_Current_City_Years': np.random.choice(['0','1','2','3','4+'], n),
            'Marital_Status': np.random.randint(0, 2, n),
            'Product_Category_1': np.random.randint(1, 19, n),
            'Product_Category_2': np.where(np.random.rand(n)>0.3, np.random.randint(1,19,n), np.nan),
            'Product_Category_3': np.where(np.random.rand(n)>0.7, np.random.randint(1,19,n), np.nan),
            'Purchase': np.random.randint(500, 24000, n),
        })
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0).astype(int)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0).astype(int)
    df.drop_duplicates(inplace=True)
    age_order = {'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7}
    df['Age_Encoded']    = df['Age'].map(age_order)
    df['Gender_Encoded'] = df['Gender'].map({'M':0,'F':1})
    df['City_Encoded']   = df['City_Category'].map({'A':1,'B':2,'C':3})
    df['Stay_Encoded']   = df['Stay_In_Current_City_Years'].replace('4+','4').astype(int)
    df['Purchase_Normalized'] = MinMaxScaler().fit_transform(df[['Purchase']])
    user_df = df.groupby('User_ID').agg(
        Total_Purchase=('Purchase','sum'), Avg_Purchase=('Purchase','mean'),
        Num_Transactions=('Purchase','count'), Age_Encoded=('Age_Encoded','first'),
        Occupation=('Occupation','first'), Marital_Status=('Marital_Status','first'),
        Gender_Encoded=('Gender_Encoded','first'), City_Encoded=('City_Encoded','first'),
        Age=('Age','first'), Gender=('Gender','first'), City_Category=('City_Category','first')
    ).reset_index()
    feat = ['Total_Purchase','Avg_Purchase','Num_Transactions','Age_Encoded','Occupation']
    X = StandardScaler().fit_transform(user_df[feat])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    user_df['Cluster'] = km.fit_predict(X)
    means = user_df.groupby('Cluster')['Total_Purchase'].mean().sort_values()
    lmap  = {means.index[0]:'Budget Shoppers', means.index[1]:'Casual Buyers',
             means.index[2]:'Regular Spenders', means.index[3]:'Premium Buyers'}
    user_df['Cluster_Label'] = user_df['Cluster'].map(lmap)
    user_spend = df.groupby('User_ID')['Purchase'].sum().reset_index()
    user_spend.columns = ['User_ID','Total_Spend']
    user_spend['ZScore'] = np.abs(stats.zscore(user_spend['Total_Spend']))
    return df, user_df, user_spend

df, user_df, user_spend = load_and_prep(uploaded)
AGE_ORDER = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']

def dark_fig(w=11, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0f0d1e'); ax.set_facecolor('#0f0d1e')
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2640')
    return fig, ax

def dark_fig2(w=13, h=5):
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.patch.set_facecolor('#0f0d1e')
    for ax in axes:
        ax.set_facecolor('#0f0d1e')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2640')
    return fig, axes

# ══════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="hero-title">Black Friday<br>Sales Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">InsightMart Analytics · Data Mining Summative · Scenario 1</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Customers",   f"{df['User_ID'].nunique():,}")
    c3.metric("Avg Purchase",       f"${df['Purchase'].mean():,.0f}")
    c4.metric("High-Spend Anomalies", f"{(user_spend['ZScore']>3).sum()}")
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Gender Split</div>', unsafe_allow_html=True)
        gc = df['Gender'].value_counts()
        fig,ax = plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#0f0d1e')
        wedges,texts,autotexts = ax.pie(gc.values, labels=['Male','Female'] if gc.index[0]=='M' else ['Female','Male'],
            autopct='%1.1f%%', colors=['#7c6af7','#ff6b35'], startangle=140,
            wedgeprops=dict(linewidth=2, edgecolor='#0f0d1e'))
        for t in texts: t.set_color('#c8c4e0')
        for t in autotexts: t.set_color('#0a0a0f'); t.set_fontweight('bold')
        ax.set_facecolor('#0f0d1e')
        st.pyplot(fig)
    with col2:
        st.markdown('<div class="section-header">City Category</div>', unsafe_allow_html=True)
        cc = df['City_Category'].value_counts()
        fig2,ax2 = dark_fig(5,4)
        bars = ax2.bar(cc.index, cc.values, color=[ACCENT,'#7c6af7','#3dcfcf'], edgecolor='#0f0d1e', linewidth=1.5, width=0.5)
        ax2.set_ylabel('Transactions', color='#7a76a0')
        for bar in bars:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                     f"{bar.get_height():,}", ha='center', color='#c8c4e0', fontsize=9, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        st.pyplot(fig2)
    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True)

elif page == "🔍  EDA & Visualizations":
    st.markdown('<div class="hero-title">Exploratory Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Uncovering patterns across demographics and product categories</div>', unsafe_allow_html=True)
    tab1,tab2,tab3,tab4 = st.tabs(["Age & Gender","Product Categories","Occupation","Correlation"])
    with tab1:
        age_means = df.groupby('Age')['Purchase'].mean().reindex(AGE_ORDER)
        fig,axes = dark_fig2(14,5)
        bars = axes[0].bar(age_means.index, age_means.values, color=PALETTE[:len(age_means)], edgecolor='#0f0d1e', linewidth=1.5)
        axes[0].set_title('Avg Purchase by Age Group', fontweight='bold', color='#e8e6f0', pad=12)
        axes[0].set_xlabel('Age Group', color='#7a76a0'); axes[0].set_ylabel('Avg Purchase (USD)', color='#7a76a0')
        axes[0].tick_params(axis='x', rotation=15, colors='#7a76a0'); axes[0].tick_params(axis='y', colors='#7a76a0')
        axes[0].grid(axis='y', alpha=0.3)
        for bar in bars:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+60,
                         f"${bar.get_height():,.0f}", ha='center', color='#c8c4e0', fontsize=7.5)
        genders = df['Gender'].unique()
        data   = [df[df['Gender']==g]['Purchase'].values for g in genders]
        labels = ['Male','Female'] if genders[0]=='M' else ['Female','Male']
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True,
                             medianprops=dict(color='#0a0a0f', linewidth=2),
                             whiskerprops=dict(color='#4a4668'), capprops=dict(color='#4a4668'),
                             flierprops=dict(marker='o', color=ACCENT, alpha=0.3, markersize=3))
        for patch,color in zip(bp['boxes'],['#7c6af7','#ff6b35']):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        axes[1].set_title('Purchase by Gender', fontweight='bold', color='#e8e6f0', pad=12)
        axes[1].set_ylabel('Purchase (USD)', color='#7a76a0'); axes[1].tick_params(colors='#7a76a0')
        axes[1].grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig)
    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            cat_counts = df['Product_Category_1'].value_counts().head(12)
            fig,ax = dark_fig(6,5)
            colors_c = [ACCENT if i==0 else '#3a3060' for i in range(len(cat_counts))]
            ax.barh(cat_counts.index.astype(str), cat_counts.values, color=colors_c, edgecolor='#0f0d1e')
            ax.invert_yaxis(); ax.set_title('Most Purchased Categories', fontweight='bold', color='#e8e6f0')
            ax.set_xlabel('Transactions', color='#7a76a0'); ax.tick_params(colors='#7a76a0'); ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
        with col2:
            cat_avg = df.groupby('Product_Category_1')['Purchase'].mean().sort_values(ascending=False).head(12)
            fig2,ax2 = dark_fig(6,5)
            colors_c2 = ['#3dcfcf' if i==0 else '#1e3a3a' for i in range(len(cat_avg))]
            ax2.barh(cat_avg.index.astype(str), cat_avg.values, color=colors_c2, edgecolor='#0f0d1e')
            ax2.invert_yaxis(); ax2.set_title('Highest Avg Spend per Category', fontweight='bold', color='#e8e6f0')
            ax2.set_xlabel('Avg Purchase ($)', color='#7a76a0'); ax2.tick_params(colors='#7a76a0'); ax2.grid(axis='x', alpha=0.3)
            st.pyplot(fig2)
    with tab3:
        occ_means = df.groupby('Occupation')['Purchase'].mean().reset_index()
        fig,ax = dark_fig(12,5)
        sc = ax.scatter(occ_means['Occupation'], occ_means['Purchase'],
                        c=occ_means['Purchase'], cmap='plasma', s=120, edgecolors='#0f0d1e', linewidth=1.5, zorder=3)
        for _,row in occ_means.iterrows():
            ax.annotate(f"${row['Purchase']:,.0f}", (row['Occupation'], row['Purchase']),
                        textcoords='offset points', xytext=(0,9), fontsize=7, ha='center', color='#9a96b8')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='#7a76a0'); cbar.outline.set_edgecolor('#2a2640')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#7a76a0')
        ax.set_xticks(occ_means['Occupation'])
        ax.set_title('Avg Purchase by Occupation', fontweight='bold', color='#e8e6f0', pad=12)
        ax.set_xlabel('Occupation', color='#7a76a0'); ax.set_ylabel('Avg Purchase ($)', color='#7a76a0')
        ax.tick_params(colors='#7a76a0'); ax.grid(alpha=0.25)
        plt.tight_layout(); st.pyplot(fig)
    with tab4:
        corr_cols = ['Age_Encoded','Occupation','City_Encoded','Stay_Encoded','Marital_Status','Product_Category_1','Purchase']
        fig,ax = plt.subplots(figsize=(9,7)); fig.patch.set_facecolor('#0f0d1e'); ax.set_facecolor('#0f0d1e')
        cmap = sns.diverging_palette(260, 20, as_cmap=True)
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap=cmap, square=True,
                    linewidths=1, linecolor='#0a0a0f', ax=ax, annot_kws={'color':'#e8e6f0','size':9})
        ax.tick_params(colors='#7a76a0', labelsize=9)
        ax.set_title('Feature Correlation Heatmap', fontweight='bold', color='#e8e6f0', pad=12)
        plt.tight_layout(); st.pyplot(fig)

elif page == "🗂️  Customer Clustering":
    st.markdown('<div class="hero-title">Customer Segments</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">K-Means (k=4) reveals 4 distinct buyer personas</div>', unsafe_allow_html=True)
    for label,color in CLUSTER_COLORS.items():
        st.markdown(f'<span class="stat-pill" style="border-color:{color};color:{color} !important;">■ {label}</span>', unsafe_allow_html=True)
    st.write("")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Clusters: Transactions vs Spend</div>', unsafe_allow_html=True)
        fig,ax = dark_fig(7,5)
        for label,color in CLUSTER_COLORS.items():
            mask = user_df['Cluster_Label']==label
            ax.scatter(user_df.loc[mask,'Num_Transactions'], user_df.loc[mask,'Total_Purchase'],
                       label=label, alpha=0.65, s=25, color=color, edgecolors='none')
        ax.set_xlabel('Num Transactions', color='#7a76a0'); ax.set_ylabel('Total Purchase ($)', color='#7a76a0')
        ax.tick_params(colors='#7a76a0'); ax.grid(alpha=0.2); ax.legend(framealpha=0.3, labelcolor='#c8c4e0', fontsize=8)
        plt.tight_layout(); st.pyplot(fig)
    with col2:
        st.markdown('<div class="section-header">Segment Sizes</div>', unsafe_allow_html=True)
        counts = user_df['Cluster_Label'].value_counts()
        fig2,ax2 = dark_fig(7,5)
        bars = ax2.bar(counts.index, counts.values, color=[CLUSTER_COLORS.get(l,'#999') for l in counts.index],
                       edgecolor='#0f0d1e', linewidth=1.5, width=0.55)
        ax2.set_ylabel('Customers', color='#7a76a0'); ax2.tick_params(axis='x', rotation=15, colors='#7a76a0')
        ax2.tick_params(axis='y', colors='#7a76a0'); ax2.grid(axis='y', alpha=0.2)
        for bar in bars:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
                     f"{int(bar.get_height()):,}", ha='center', color='#c8c4e0', fontsize=9, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig2)
    st.markdown('<div class="section-header">Cluster Summary</div>', unsafe_allow_html=True)
    feat = ['Total_Purchase','Avg_Purchase','Num_Transactions','Age_Encoded','Occupation']
    st.dataframe(user_df.groupby('Cluster_Label')[feat].mean().round(2), use_container_width=True)
    st.markdown('<div class="section-header">Demographics per Cluster</div>', unsafe_allow_html=True)
    col3,col4 = st.columns(2)
    with col3:
        g_mix = user_df.groupby(['Cluster_Label','Gender']).size().unstack(fill_value=0)
        fig3,ax3 = dark_fig(6.5,4)
        x = np.arange(len(g_mix)); w = 0.35
        ax3.bar(x-w/2, g_mix.get('M', pd.Series([0]*len(g_mix))), w, label='Male', color='#7c6af7', edgecolor='#0f0d1e')
        ax3.bar(x+w/2, g_mix.get('F', pd.Series([0]*len(g_mix))), w, label='Female', color='#ff6b35', edgecolor='#0f0d1e')
        ax3.set_xticks(x); ax3.set_xticklabels(g_mix.index, rotation=12, ha='right', color='#7a76a0', fontsize=8)
        ax3.tick_params(axis='y', colors='#7a76a0'); ax3.set_title('Gender per Cluster', fontweight='bold', color='#e8e6f0')
        ax3.legend(labelcolor='#c8c4e0', framealpha=0.3, fontsize=8); ax3.grid(axis='y', alpha=0.2)
        plt.tight_layout(); st.pyplot(fig3)
    with col4:
        a_mix = user_df.groupby(['Cluster_Label','Age']).size().unstack(fill_value=0)
        a_mix = a_mix.reindex(columns=[c for c in AGE_ORDER if c in a_mix.columns])
        fig4,ax4 = dark_fig(6.5,4)
        a_mix.plot(kind='bar', ax=ax4, color=PALETTE, edgecolor='#0f0d1e', linewidth=0.5)
        ax4.set_title('Age Groups per Cluster', fontweight='bold', color='#e8e6f0'); ax4.set_xlabel('', color='#7a76a0')
        ax4.tick_params(axis='x', rotation=12, colors='#7a76a0', labelsize=8); ax4.tick_params(axis='y', colors='#7a76a0')
        ax4.legend(fontsize=7, loc='upper right', labelcolor='#c8c4e0', framealpha=0.3); ax4.grid(axis='y', alpha=0.2)
        plt.tight_layout(); st.pyplot(fig4)

elif page == "🔗  Association Rules":
    st.markdown('<div class="hero-title">Association Rules</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Product categories frequently purchased together</div>', unsafe_allow_html=True)
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
        def make_transaction(row):
            cats = [f"Cat_{int(row['Product_Category_1'])}"]
            if row['Product_Category_2']!=0: cats.append(f"Cat_{int(row['Product_Category_2'])}")
            if row['Product_Category_3']!=0: cats.append(f"Cat_{int(row['Product_Category_3'])}")
            return list(set(cats))
        col1,col2 = st.columns(2)
        min_sup  = col1.slider("Min Support",    0.01, 0.3, 0.05, 0.01)
        min_conf = col2.slider("Min Confidence", 0.2,  0.9, 0.4,  0.05)
        with st.spinner("⚡ Mining association rules..."):
            transactions = df.apply(make_transaction, axis=1).tolist()
            te = TransactionEncoder()
            te_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
            fi = apriori(te_df, min_support=min_sup, use_colnames=True)
        if len(fi)==0:
            st.warning("No itemsets found. Try lowering min support.")
        else:
            rules = association_rules(fi, metric='confidence', min_threshold=min_conf)
            rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            rules['rule'] = rules['antecedents'] + '  →  ' + rules['consequents']
            c1,c2,c3 = st.columns(3)
            c1.metric("Rules Generated", len(rules)); c2.metric("Frequent Itemsets", len(fi)); c3.metric("Max Lift", f"{rules['lift'].max():.2f}")
            st.markdown('<div class="section-header">Top Rules by Lift</div>', unsafe_allow_html=True)
            top = rules.head(12); fig,ax = dark_fig(12,5)
            colors_r = [ACCENT if i<3 else '#3a3060' for i in range(len(top))]
            ax.barh(top['rule'], top['lift'], color=colors_r, edgecolor='#0f0d1e', linewidth=1)
            ax.invert_yaxis(); ax.set_xlabel('Lift', color='#7a76a0')
            ax.tick_params(axis='y', colors='#c8c4e0', labelsize=8); ax.tick_params(axis='x', colors='#7a76a0'); ax.grid(axis='x', alpha=0.25)
            for bar in ax.patches[:3]:
                ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                        f"{bar.get_width():.2f}", va='center', color=ACCENT, fontsize=8, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig)
            st.markdown('<div class="section-header">Rules Table</div>', unsafe_allow_html=True)
            st.dataframe(rules[['rule','support','confidence','lift']].head(15)
                         .style.format({'support':'{:.3f}','confidence':'{:.3f}','lift':'{:.3f}'}), use_container_width=True)
    except ImportError:
        st.error("mlxtend not installed. Add it to requirements.txt.")

elif page == "🚨  Anomaly Detection":
    st.markdown('<div class="hero-title">Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Identifying unusually high-spending customers via Z-Score and IQR</div>', unsafe_allow_html=True)
    z_thresh = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.5)
    anomalies = user_spend[user_spend['ZScore']>z_thresh]; normal = user_spend[user_spend['ZScore']<=z_thresh]
    Q1 = user_spend['Total_Spend'].quantile(0.25); Q3 = user_spend['Total_Spend'].quantile(0.75)
    IQR = Q3-Q1; upper = Q3+1.5*IQR
    c1,c2,c3 = st.columns(3)
    c1.metric("Anomalous Customers", len(anomalies), delta="VIP targets")
    c2.metric("Normal Customers", len(normal)); c3.metric("IQR Upper Bound", f"${upper:,.0f}")
    st.markdown('<div class="section-header">Spend Distribution</div>', unsafe_allow_html=True)
    fig,ax = dark_fig(12,5)
    ax.scatter(range(len(normal)), normal['Total_Spend'], color='#3a3060', alpha=0.5, s=12, label='Normal')
    ax.scatter(anomalies.index, anomalies['Total_Spend'], color=ACCENT, s=70, zorder=5,
               label=f'Anomaly ({len(anomalies)})', edgecolors='#ffb38a', linewidth=0.8)
    ax.axhline(upper, color='#f7c559', linestyle='--', linewidth=1.5, label=f'IQR Upper ${upper:,.0f}')
    ax.set_xlabel('Customer Index', color='#7a76a0'); ax.set_ylabel('Total Spend ($)', color='#7a76a0')
    ax.tick_params(colors='#7a76a0'); ax.grid(alpha=0.2); ax.legend(framealpha=0.3, labelcolor='#c8c4e0')
    plt.tight_layout(); st.pyplot(fig)
    if len(anomalies)>0:
        st.markdown('<div class="section-header">Anomalous Customer Profiles</div>', unsafe_allow_html=True)
        profile = anomalies.merge(df[['User_ID','Age','Gender','Occupation','City_Category']].drop_duplicates('User_ID'), on='User_ID').sort_values('Total_Spend', ascending=False)
        st.dataframe(profile.head(20).round(2), use_container_width=True)

elif page == "💡  Key Insights":
    st.markdown('<div class="hero-title">Key Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Strategic findings from Black Friday transaction analysis</div>', unsafe_allow_html=True)
    top_age      = df.groupby('Age')['Purchase'].mean().idxmax()
    top_age_val  = df.groupby('Age')['Purchase'].mean().max()
    gender_spend = df.groupby('Gender')['Purchase'].mean()
    top_cat      = df['Product_Category_1'].value_counts().idxmax()
    city_spend   = df.groupby('City_Category')['Purchase'].mean()
    n_anomalies  = (user_spend['ZScore']>3).sum()
    insights = [
        ("🎯 Prime Demographic",
         f"The <strong style='color:#ff9a6b'>{top_age}</strong> age group leads with avg spend of <strong style='color:#ff9a6b'>${top_age_val:,.0f}</strong>/transaction."),
        ("👥 Gender Spending Gap",
         f"Male customers avg <strong style='color:#7c6af7'>${gender_spend.get('M',0):,.0f}</strong> vs Female <strong style='color:#ff6b35'>${gender_spend.get('F',0):,.0f}</strong>. Males dominate Black Friday spend."),
        ("🛒 Star Category",
         f"Product Category <strong style='color:#3dcfcf'>{top_cat}</strong> is most purchased. Bundle and cross-sell around this for max revenue."),
        ("🏙️ City Insights",
         f"City B leads in avg spend at <strong style='color:#f7c559'>${city_spend.get('B',0):,.0f}</strong>. Prioritize promotional budgets in City B."),
        ("⚠️ VIP Anomalies",
         f"<strong style='color:#ff6b35'>{n_anomalies} high-spending anomalies</strong> detected — offer exclusive loyalty rewards to convert them."),
        ("🗂️ Four Buyer Personas",
         "Clustering reveals: <strong style='color:#f7c559'>Budget Shoppers</strong>, <strong style='color:#7c6af7'>Casual Buyers</strong>, <strong style='color:#3dcfcf'>Regular Spenders</strong>, and <strong style='color:#ff6b35'>Premium Buyers</strong> — each needs distinct strategy."),
        ("🔗 Cross-Sell Wins",
         "Association rules reveal strong co-purchase patterns. <strong style='color:#ff9a6b'>Bundle deals</strong> around these combinations lift basket size significantly."),
    ]
    for i,(title,body) in enumerate(insights):
        st.markdown(f'<div class="insight-card" style="animation-delay:{i*0.07}s"><strong style="font-size:1rem;font-family:Syne,sans-serif;">{title}</strong><p style="margin-top:6px;">{body}</p></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-header">Summary Dashboard</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Transactions", f"{len(df):,}"); c2.metric("Unique Customers", f"{df['User_ID'].nunique():,}"); c3.metric("Avg Transaction", f"${df['Purchase'].mean():,.0f}")
    c4,c5,c6 = st.columns(3)
    c4.metric("Top Age Group", top_age); c5.metric("Top Product Cat", f"Category {top_cat}"); c6.metric("VIP Buyers Flagged", f"{n_anomalies}")
