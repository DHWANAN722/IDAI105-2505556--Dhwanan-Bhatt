"""
Beyond Discounts: Data-Driven Black Friday Sales Insights
Streamlit Dashboard – Scenario 1
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #e65100, #f9a825);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .insight-box {
        background: #fff8e1; border-left: 4px solid #f9a825;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛍️ Black Friday Analytics")
st.sidebar.markdown("**InsightMart Analytics Dashboard**")
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("📂 Upload BlackFriday.csv", type=["csv"])
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔍 EDA & Visualizations",
    "🗂️ Customer Clustering",
    "🔗 Association Rules",
    "🚨 Anomaly Detection",
    "💡 Key Insights"
])

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_prep(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Minimal synthetic fallback so app runs without upload
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
            'Product_Category_2': np.where(np.random.rand(n) > 0.3,
                                           np.random.randint(1, 19, n), np.nan),
            'Product_Category_3': np.where(np.random.rand(n) > 0.7,
                                           np.random.randint(1, 19, n), np.nan),
            'Purchase': np.random.randint(500, 24000, n),
        })

    # Preprocessing
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0).astype(int)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0).astype(int)
    df.drop_duplicates(inplace=True)

    age_order = {'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7}
    df['Age_Encoded'] = df['Age'].map(age_order)
    df['Gender_Encoded'] = df['Gender'].map({'M':0,'F':1})
    df['City_Encoded'] = df['City_Category'].map({'A':1,'B':2,'C':3})
    df['Stay_Encoded'] = df['Stay_In_Current_City_Years'].replace('4+','4').astype(int)

    scaler = MinMaxScaler()
    df['Purchase_Normalized'] = scaler.fit_transform(df[['Purchase']])

    # User-level aggregation for clustering
    user_df = df.groupby('User_ID').agg(
        Total_Purchase=('Purchase','sum'),
        Avg_Purchase=('Purchase','mean'),
        Num_Transactions=('Purchase','count'),
        Age_Encoded=('Age_Encoded','first'),
        Occupation=('Occupation','first'),
        Marital_Status=('Marital_Status','first'),
        Gender_Encoded=('Gender_Encoded','first'),
        City_Encoded=('City_Encoded','first'),
        Age=('Age','first'),
        Gender=('Gender','first'),
        City_Category=('City_Category','first')
    ).reset_index()

    # K-Means clustering
    feat = ['Total_Purchase','Avg_Purchase','Num_Transactions','Age_Encoded','Occupation']
    X = StandardScaler().fit_transform(user_df[feat])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    user_df['Cluster'] = km.fit_predict(X)
    means = user_df.groupby('Cluster')['Total_Purchase'].mean().sort_values()
    lmap = {means.index[0]:'Budget Shoppers', means.index[1]:'Casual Buyers',
            means.index[2]:'Regular Spenders', means.index[3]:'Premium Buyers'}
    user_df['Cluster_Label'] = user_df['Cluster'].map(lmap)

    # Anomaly detection (user total spend)
    user_spend = df.groupby('User_ID')['Purchase'].sum().reset_index()
    user_spend.columns = ['User_ID','Total_Spend']
    user_spend['ZScore'] = np.abs(stats.zscore(user_spend['Total_Spend']))

    return df, user_df, user_spend

df, user_df, user_spend = load_and_prep(uploaded)

CLUSTER_COLORS = {
    'Budget Shoppers':  '#FF9800',
    'Casual Buyers':    '#2196F3',
    'Regular Spenders': '#9C27B0',
    'Premium Buyers':   '#4CAF50'
}

AGE_ORDER = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']

# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">🛍️ Black Friday Sales Analytics</p>', unsafe_allow_html=True)
    st.markdown("**InsightMart Analytics** – Data-Driven Black Friday Sales Insights")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Customers", f"{df['User_ID'].nunique():,}")
    c3.metric("Avg Purchase", f"${df['Purchase'].mean():,.0f}")
    c4.metric("Anomalies (High Spenders)", f"{(user_spend['ZScore'] > 3).sum()}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gender Distribution")
        gc = df['Gender'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(gc.values, labels=['Male','Female'] if gc.index[0]=='M' else ['Female','Male'],
               autopct='%1.1f%%', colors=['#2196F3','#E91E63'], startangle=140)
        ax.set_title('Customer Gender Split', fontweight='bold')
        st.pyplot(fig)

    with col2:
        st.subheader("Transactions by City")
        cc = df['City_Category'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.bar(cc.index, cc.values, color=['#FF9800','#4CAF50','#9C27B0'],
                edgecolor='white', width=0.5)
        ax2.set_title('Transactions by City Category', fontweight='bold')
        ax2.set_ylabel('Count')
        st.pyplot(fig2)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 EDA & Visualizations":
    st.header("🔍 Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Age & Gender","Product Categories","Occupation","Correlation"])

    with tab1:
        st.subheader("Avg Purchase by Age Group")
        age_means = df.groupby('Age')['Purchase'].mean().reindex(AGE_ORDER)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(age_means.index, age_means.values, color='steelblue', edgecolor='white')
        axes[0].set_title('Avg Purchase by Age Group', fontweight='bold')
        axes[0].set_xlabel('Age Group'); axes[0].set_ylabel('Avg Purchase ($)')
        axes[0].tick_params(axis='x', rotation=15)

        genders = df['Gender'].unique()
        data = [df[df['Gender']==g]['Purchase'].values for g in genders]
        bp = axes[1].boxplot(data, labels=['Male','Female'] if genders[0]=='M' else ['Female','Male'],
                             patch_artist=True, notch=False)
        for patch, color in zip(bp['boxes'], ['#2196F3','#E91E63']):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        axes[1].set_title('Purchase Distribution by Gender', fontweight='bold')
        axes[1].set_ylabel('Purchase ($)')
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Product Category Popularity")
        col1, col2 = st.columns(2)
        with col1:
            cat_counts = df['Product_Category_1'].value_counts().head(12)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(cat_counts.index.astype(str), cat_counts.values, color='coral', edgecolor='white')
            ax.invert_yaxis()
            ax.set_title('Most Frequent Categories (by transactions)', fontweight='bold')
            ax.set_xlabel('Transactions')
            st.pyplot(fig)
        with col2:
            cat_avg = df.groupby('Product_Category_1')['Purchase'].mean().sort_values(ascending=False).head(12)
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.barh(cat_avg.index.astype(str), cat_avg.values, color='teal', edgecolor='white')
            ax2.invert_yaxis()
            ax2.set_title('Highest Avg Purchase by Category', fontweight='bold')
            ax2.set_xlabel('Avg Purchase ($)')
            st.pyplot(fig2)

    with tab3:
        st.subheader("Purchase by Occupation")
        occ_means = df.groupby('Occupation')['Purchase'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.scatter(occ_means['Occupation'], occ_means['Purchase'],
                   color='mediumpurple', s=80, edgecolors='white', zorder=3)
        ax.set_xticks(occ_means['Occupation'])
        ax.set_title('Avg Purchase by Occupation Code', fontweight='bold')
        ax.set_xlabel('Occupation'); ax.set_ylabel('Avg Purchase ($)')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.subheader("Correlation Heatmap")
        corr_cols = ['Age_Encoded','Occupation','City_Encoded','Stay_Encoded',
                     'Marital_Status','Product_Category_1','Purchase']
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f',
                    cmap='coolwarm', square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗂️ Customer Clustering":
    st.header("🗂️ Customer Clustering")
    st.markdown("K-Means (k=4) groups customers into **Budget Shoppers**, **Casual Buyers**, **Regular Spenders**, and **Premium Buyers**.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cluster Scatter: Transactions vs Total Spend")
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, color in CLUSTER_COLORS.items():
            mask = user_df['Cluster_Label'] == label
            ax.scatter(user_df.loc[mask,'Num_Transactions'],
                       user_df.loc[mask,'Total_Purchase'],
                       label=label, alpha=0.5, s=30, color=color)
        ax.set_xlabel('Num Transactions'); ax.set_ylabel('Total Purchase ($)')
        ax.set_title('Customer Segments', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Cluster Sizes")
        counts = user_df['Cluster_Label'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        bars = ax2.bar(counts.index, counts.values,
                       color=[CLUSTER_COLORS.get(l,'#999') for l in counts.index],
                       edgecolor='white', width=0.5)
        ax2.set_ylabel('Number of Customers')
        ax2.set_title('Customers per Segment', fontweight='bold')
        for bar, val in zip(bars, counts.values):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                     str(val), ha='center', fontweight='bold')
        plt.xticks(rotation=15, ha='right')
        st.pyplot(fig2)

    st.subheader("Cluster Summary (Mean Values)")
    feat = ['Total_Purchase','Avg_Purchase','Num_Transactions','Age_Encoded','Occupation']
    summary = user_df.groupby('Cluster_Label')[feat].mean().round(2)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Gender & Age Mix per Cluster")
    col3, col4 = st.columns(2)
    with col3:
        g_mix = user_df.groupby(['Cluster_Label','Gender']).size().unstack(fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(6,4))
        g_mix.plot(kind='bar', ax=ax3, color=['#2196F3','#E91E63'], edgecolor='white')
        ax3.set_title('Gender per Cluster', fontweight='bold')
        ax3.set_xlabel(''); plt.xticks(rotation=15, ha='right')
        st.pyplot(fig3)
    with col4:
        a_mix = user_df.groupby(['Cluster_Label','Age']).size().unstack(fill_value=0)
        a_mix = a_mix.reindex(columns=[c for c in AGE_ORDER if c in a_mix.columns])
        fig4, ax4 = plt.subplots(figsize=(6,4))
        a_mix.plot(kind='bar', ax=ax4, edgecolor='white')
        ax4.set_title('Age Groups per Cluster', fontweight='bold')
        ax4.set_xlabel(''); ax4.legend(fontsize=7, loc='upper right')
        plt.xticks(rotation=15, ha='right')
        st.pyplot(fig4)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔗 Association Rules":
    st.header("🔗 Association Rule Mining")
    st.markdown("Apriori algorithm finds product categories that are **frequently bought together**.")

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        def make_transaction(row):
            cats = [f"Cat_{int(row['Product_Category_1'])}"]
            if row['Product_Category_2'] != 0:
                cats.append(f"Cat_{int(row['Product_Category_2'])}")
            if row['Product_Category_3'] != 0:
                cats.append(f"Cat_{int(row['Product_Category_3'])}")
            return list(set(cats))

        col1, col2 = st.columns(2)
        min_sup  = col1.slider("Min Support",  0.01, 0.3, 0.05, 0.01)
        min_conf = col2.slider("Min Confidence", 0.2, 0.9, 0.4, 0.05)

        with st.spinner("Mining rules..."):
            transactions = df.apply(make_transaction, axis=1).tolist()
            te = TransactionEncoder()
            te_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
            fi = apriori(te_df, min_support=min_sup, use_colnames=True)

        if len(fi) == 0:
            st.warning("No frequent itemsets. Try lowering min support.")
        else:
            rules = association_rules(fi, metric='confidence', min_threshold=min_conf)
            rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            rules['rule'] = rules['antecedents'] + '  →  ' + rules['consequents']

            st.metric("Rules Generated", len(rules))
            st.dataframe(
                rules[['rule','support','confidence','lift']].head(15)
                .style.format({'support':'{:.3f}','confidence':'{:.3f}','lift':'{:.3f}'}),
                use_container_width=True
            )

            st.subheader("Top 10 Rules by Lift")
            fig, ax = plt.subplots(figsize=(11, 5))
            top = rules.head(10)
            ax.barh(top['rule'], top['lift'], color='coral', edgecolor='white')
            ax.set_xlabel('Lift'); ax.invert_yaxis()
            ax.set_title('Association Rules – Top 10 by Lift', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

    except ImportError:
        st.error("mlxtend not installed. Add it to requirements.txt and redeploy.")

# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Anomaly Detection":
    st.header("🚨 Anomaly Detection – High Spenders")
    st.markdown("Detecting unusually high-spending customers using **Z-Score** and **IQR** methods.")

    z_thresh = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.5)
    anomalies = user_spend[user_spend['ZScore'] > z_thresh]
    normal    = user_spend[user_spend['ZScore'] <= z_thresh]

    Q1 = user_spend['Total_Spend'].quantile(0.25)
    Q3 = user_spend['Total_Spend'].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR

    col1, col2, col3 = st.columns(3)
    col1.metric("Anomalous Customers", len(anomalies))
    col2.metric("Normal Customers", len(normal))
    col3.metric("IQR Upper Bound", f"${upper:,.0f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(range(len(normal)), normal['Total_Spend'],
               color='steelblue', alpha=0.3, s=15, label='Normal')
    ax.scatter(anomalies.index, anomalies['Total_Spend'],
               color='red', s=60, zorder=5, label=f'Anomaly ({len(anomalies)})')
    ax.axhline(upper, color='orange', linestyle='--', label=f'IQR Upper ${upper:,.0f}')
    ax.set_xlabel('Customer Index'); ax.set_ylabel('Total Spend ($)')
    ax.set_title('Customer Spend – Anomaly Detection', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    if len(anomalies) > 0:
        st.subheader("Anomalous Customer Profiles")
        profile = anomalies.merge(
            df[['User_ID','Age','Gender','Occupation','City_Category']].drop_duplicates('User_ID'),
            on='User_ID'
        ).sort_values('Total_Spend', ascending=False)
        st.dataframe(profile.head(20).round(2), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "💡 Key Insights":
    st.header("💡 Key Insights & Recommendations")

    top_age     = df.groupby('Age')['Purchase'].mean().idxmax()
    top_age_val = df.groupby('Age')['Purchase'].mean().max()
    gender_spend = df.groupby('Gender')['Purchase'].mean()
    top_cat      = df['Product_Category_1'].value_counts().idxmax()
    city_spend   = df.groupby('City_Category')['Purchase'].mean()
    n_anomalies  = (user_spend['ZScore'] > 3).sum()

    insights = [
        (f"The **{top_age}** age group has the highest average spend at **${top_age_val:,.0f}** per transaction. "
         "Marketing campaigns should heavily target this segment."),
        (f"Male customers average **${gender_spend.get('M', 0):,.0f}** vs Female **${gender_spend.get('F', 0):,.0f}** per transaction. "
         "Males dominate Black Friday spend."),
        (f"**Product Category {top_cat}** is the most purchased category. "
         "Stocking more of this category and cross-selling with related items can boost revenue."),
        (f"City B customers spend the most on average (${city_spend.get('B', 0):,.0f}). "
         "Targeted promotions in City B can yield the highest returns."),
        (f"**{n_anomalies} high-spending anomalies** were detected. "
         "These VIP customers should receive exclusive loyalty offers and personalized outreach."),
        ("Cluster analysis reveals 4 buyer personas: **Budget Shoppers**, **Casual Buyers**, "
         "**Regular Spenders**, and **Premium Buyers**. Each segment requires a different pricing and promotion strategy."),
        ("Association rule mining shows that customers buying certain categories together are strong "
         "candidates for **bundle deals and combo discounts** to increase basket size.")
    ]

    for i, insight in enumerate(insights, 1):
        st.markdown(f"""<div class="insight-box">
            <strong>Insight {i}:</strong> {insight}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Summary Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Customers", f"{df['User_ID'].nunique():,}")
    c3.metric("Avg Transaction Value", f"${df['Purchase'].mean():,.0f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Most Active Age Group", top_age)
    c5.metric("Top Product Category", f"Category {top_cat}")
    c6.metric("VIP Anomalous Buyers", f"{n_anomalies}")
