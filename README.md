# ⚡ BLACK FRIDAY // DATA CORE
### Mining the Future: Black Friday Sales Insights
**Data Mining – Year 1 Summative Assessment | CRS: Artificial Intelligence**


APP LINK: https://idai105-2505556--dhwanan-bhatt-zhgnhej6sxvysynfud5doz.streamlit.app/

## 🧠 Project Overview

**BLACK FRIDAY // DATA CORE** is a full-stack data mining and business intelligence system built for InsightMart Analytics. It analyzes over 550,000 Black Friday retail transactions to extract deep, actionable insights about customer shopping behavior. The project covers the complete data science pipeline — from raw data ingestion and preprocessing, through exploratory analysis, unsupervised machine learning, association rule mining, statistical anomaly detection, and finally a fully deployed interactive dashboard on Streamlit Cloud.

The driving question behind this system is simple but powerful: inside half a million transactions, what stories are hiding? Who are the biggest spenders? Which products always get bought together? Are there unusual transactions that look nothing like the rest? This project answers all of it, end to end.

---

## 📦 Dataset

The **Black Friday Sales Dataset** contains transactional records from a retail chain's Black Friday event. Each row represents a single purchase made by a customer.

| Column | Description |
|---|---|
| `User_ID` | Unique identifier for each customer |
| `Product_ID` | Unique identifier for each product |
| `Gender` | Customer gender — M (Male) or F (Female) |
| `Age` | Age group bracket — 0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55+ |
| `Occupation` | Occupation code from 0 to 20 |
| `City_Category` | City tier — A (Metro), B (Tier-2), C (Tier-3) |
| `Stay_In_Current_City_Years` | How long the customer has lived in current city |
| `Marital_Status` | 0 = Single, 1 = Married |
| `Product_Category_1` | Primary product category code |
| `Product_Category_2` | Secondary product category (often missing) |
| `Product_Category_3` | Tertiary product category (often missing) |
| `Purchase` | Purchase amount in USD — the target variable |

**Dataset size:** ~550,000 rows | ~537,000 after deduplication

---

## ⚙️ Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.14 | Core runtime |
| Dashboard | Streamlit 1.55 | Interactive web UI |
| Data Processing | Pandas 2.3, NumPy 2.4 | Cleaning, transformation, aggregation |
| Machine Learning | Scikit-learn 1.8 | KMeans clustering, PCA, StandardScaler |
| Association Mining | MLxtend 0.24 | Apriori algorithm, TransactionEncoder |
| Statistical Analysis | SciPy 1.17 | Z-score calculation, IQR outlier detection |
| Visualization | Matplotlib 3.10, Seaborn 0.13 | All charts and heatmaps |
| Deployment | Streamlit Cloud | Live public web app |

---

## 🔬 Code Deep-Dive — What Every Part Does

### 1 · Data Loading & Caching (`@st.cache_data`)

```python
@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)
    ...
```

The `@st.cache_data` decorator is critical for performance. Without it, Streamlit re-runs the entire data loading and cleaning pipeline every time the user interacts with any widget — which on 550,000 rows would take several seconds each time. With caching, the cleaned dataframe is stored in memory after the first load and reused instantly for all subsequent interactions.

### 2 · Data Cleaning & Preprocessing

```python
df["Product_Category_2"] = df["Product_Category_2"].fillna(0).astype(int)
df["Product_Category_3"] = df["Product_Category_3"].fillna(0).astype(int)
df = df.drop_duplicates().reset_index(drop=True)
```

`Product_Category_2` and `Product_Category_3` have significant missing values because many transactions only involve a single product category. These nulls are filled with `0` rather than dropped, because `0` has a meaningful interpretation here — the customer simply did not purchase in a secondary category. Dropping those rows entirely would remove a large portion of valid transactions.

```python
df["Gender_Encoded"] = df["Gender"].map({"M": 0, "F": 1})
age_map = {"0-17":1,"18-25":2,"26-35":3,"36-45":4,"46-50":5,"51-55":6,"55+":7}
df["Age_Encoded"] = df["Age"].map(age_map)
df["Stay_Encoded"] = df["Stay_In_Current_City_Years"].replace("4+", 4).astype(int)
```

Machine learning algorithms require numerical inputs. Categorical columns like Gender and Age are encoded into ordered integers. The `"4+"` value in stay years is a string edge case that needs explicit handling before the column can be cast to integer.

```python
df["Purchase_Normalized"] = StandardScaler().fit_transform(df[["Purchase"]])
```

StandardScaler transforms Purchase amounts to have mean=0 and standard deviation=1. This is essential before clustering — without normalization, Purchase (ranging into thousands) would completely dominate the distance calculations over small-scale features like Age_Encoded (ranging 1-7).

---

## 🧬 Clustering Analysis — Full Technical Breakdown

Clustering is the heart of the analytical work in this project. The goal is to group customers into meaningful segments based on their shopping behavior, without using any predefined labels. This is **unsupervised machine learning**.

### Feature Engineering for Clustering

Raw transaction-level data cannot be fed directly into a clustering algorithm because each customer appears in hundreds of rows. First, the data is aggregated to the user level:

```python
user_df = df.groupby("User_ID").agg(
    Age_Encoded=("Age_Encoded", "first"),
    Gender_Encoded=("Gender_Encoded", "first"),
    Occupation=("Occupation", "first"),
    Marital_Status=("Marital_Status", "first"),
    Avg_Purchase=("Purchase", "mean"),
    Total_Purchase=("Purchase", "sum"),
    Num_Transactions=("Purchase", "count")
).reset_index()
```

This produces one row per customer, with features that describe their overall profile: demographics plus spending behavior. `Avg_Purchase` captures typical spend per transaction, while `Num_Transactions` captures how frequently they shop.

### Feature Scaling

```python
features = ["Age_Encoded","Gender_Encoded","Occupation",
            "Marital_Status","Avg_Purchase","Num_Transactions"]
X = StandardScaler().fit_transform(user_df[features])
```

All six features are standardized before clustering. This ensures that `Avg_Purchase` and `Num_Transactions` — which have large absolute values — don't overwhelm demographic features in the Euclidean distance calculations that KMeans relies on.

### K-Means Algorithm

K-Means works by assigning each data point to the nearest cluster centroid, then recalculating centroids as the mean of all assigned points, and repeating until convergence. It minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called inertia:

```
WCSS = Σ Σ ||x_i - μ_k||²
```

Where `x_i` is a data point and `μ_k` is its cluster's centroid.

```python
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
user_df["Cluster"] = kmeans.fit_predict(X)
```

`random_state=42` ensures reproducibility. `n_init=10` means the algorithm runs 10 times with different random centroid initializations and keeps the best result — this guards against getting stuck in a poor local minimum.

### Elbow Method — Choosing k

```python
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
```

The Elbow Method plots inertia against number of clusters. As k increases, inertia always decreases — but at some point the improvement becomes marginal. The "elbow" in the curve marks the optimal k where adding more clusters stops being meaningfully useful. For this dataset, k=4 is the elbow point, giving four well-separated customer segments.

### PCA Visualization

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

The clustering is performed in 6-dimensional feature space, which cannot be visualized directly. Principal Component Analysis (PCA) reduces this to 2 dimensions by finding the directions of maximum variance in the data. The resulting scatter plot lets us visually verify that the four clusters are genuinely distinct and well-separated in the data.

### Cluster Labels

After clustering, each segment is labeled based on its average purchase amount:

| Segment | Behavior |
|---|---|
| 🟦 BUDGET UNIT | Low average spend, fewer transactions |
| 🟩 CASUAL NODE | Moderate spend, occasional shopping |
| 🟧 REGULAR GRID | Above-average spend, frequent buyer |
| 🟥 PREMIUM CORE | High spend, high transaction volume |

---

## 🔮 Association Rule Mining — Full Technical Breakdown

Association rule mining discovers which products tend to be purchased together. This is the algorithm behind "customers who bought X also bought Y."

### Building the Transaction Basket

```python
basket = df.groupby("User_ID").apply(
    lambda x: list(set(
        ["Cat1_"+str(int(c)) for c in x["Product_Category_1"] if c!=0] +
        ["Cat2_"+str(int(c)) for c in x["Product_Category_2"] if c!=0] +
        ["Cat3_"+str(int(c)) for c in x["Product_Category_3"] if c!=0]
    ))
).tolist()
```

Each user's complete purchase history across all three category columns is collapsed into a single set of items — their "basket." The `set()` removes duplicates so each category only counts once per user.

### Apriori Algorithm

```python
frequent_itemsets = apriori(basket_df, min_support=0.10, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2, num_itemsets=len(frequent_itemsets))
```

The Apriori algorithm works in two phases:
1. **Find frequent itemsets** — combinations of items that appear in at least `min_support` fraction of all transactions
2. **Generate rules** — for each frequent itemset, derive if-then rules that meet the confidence and lift thresholds

### Key Metrics

**Support** — How often the itemset appears in all transactions:
```
Support(A→B) = transactions containing both A and B / total transactions
```

**Confidence** — How often B appears given A was purchased:
```
Confidence(A→B) = transactions containing both A and B / transactions containing A
```

**Lift** — How much more likely B is given A, compared to B appearing randomly:
```
Lift(A→B) = Confidence(A→B) / Support(B)
```
A lift > 1 means a positive association. Lift = 1 means independence. The dashboard filters for lift ≥ 1.2, meaning the co-purchase is at least 20% more likely than chance.

---

## ☢️ Anomaly Detection — Full Technical Breakdown

### Z-Score Method

```python
df["Z_Score"] = np.abs(stats.zscore(df["Purchase"]))
df["Is_Anomaly"] = df["Z_Score"] > 3
```

The Z-score measures how many standard deviations a value is from the mean:
```
Z = (x - μ) / σ
```
Values with |Z| > 3 are flagged as anomalies. In a normal distribution, only 0.3% of values fall beyond ±3 standard deviations, so these are genuinely extreme transactions. The threshold is adjustable in the dashboard via a slider.

### IQR Method

The Interquartile Range provides a second, more robust outlier boundary:
```
Upper Bound = Q3 + 1.5 × (Q3 - Q1)
```
This is plotted as a reference line on the distribution histogram and is less sensitive to extreme values than Z-score.

---

## 🎨 The Dashboard — App Walkthrough

The live app at **[BLACK FRIDAY // DATA CORE](https://idai105-2505556--dhwanan-bhatt-avvvz9hkwksrdi73fa5umb.streamlit.app/)** is built with a full cyberpunk neo-noir aesthetic — deep void-black backgrounds, neon cyan/pink/yellow color palette, Orbitron display font, and Share Tech Mono for all data readouts. Every chart uses a dark background with glowing neon bars.

### 📡 EDA SCAN Tab
Four headline metrics at the top (total transactions, unique customers, unique products, average spend). Below that: age cohort spend chart, gender spend split, top product category frequency, average revenue per category, and a full 10×10 feature correlation heatmap using a custom diverging colormap.

### 🧬 CLUSTER MAP Tab
Elbow curve with the optimal k=4 marked, segment distribution pie chart, PCA scatter plot with four color-coded clusters rendered with glow effects, and a statistics table showing avg spend, total spend, transaction count, and user count per segment.

### 🔮 RULE ENGINE Tab
Two interactive sliders — minimum support and minimum lift — that re-run the Apriori algorithm live. Results shown as a horizontal bar chart of top 15 rules ranked by lift, plus a full scrollable rules table with support, confidence, and lift values.

### ☢️ ANOMALY TRACE Tab
Z-score threshold slider that dynamically updates all anomaly counts. Scatter plot showing normal transactions in cyan and anomalies in glowing pink. Purchase distribution histogram with IQR upper boundary. Breakdown table of anomalies by age cohort and a sample flagged transactions table.

### 💀 INTEL REPORT Tab
Styled neon stat cards for every key finding. Strategic recommendations rendered as a terminal command-style readout with numbered intel items. All insights derived directly from the analysis results.

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/YourUsername/IDAI105-2505556--dhwanan-bhatt
cd IDAI105-2505556--dhwanan-bhatt
pip install -r requirements.txt
streamlit run app.py
```
Upload `train.csv` via the sidebar when the app opens in your browser.

---

## 📁 Repository Structure

```
├── app.py                  # Main Streamlit dashboard — cyberpunk UI, all 5 tabs
├── analysis.ipynb          # Full Jupyter Notebook — all 8 pipeline stages
├── requirements.txt        # Python dependencies (no pinned versions)
├── train.csv               # Black Friday dataset (~550k rows)
└── README.md               # This file
```

---

## 📚 References

- [K-Means Clustering — Neptune.ai](https://neptune.ai/blog/k-means-clustering)
- [Association Rule Mining — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-market-basket-analysis/)
- [Anomaly Detection — DataCamp](https://www.datacamp.com/courses/anomaly-detection-in-python)
- [Scikit-learn: Clustering & Anomaly Detection](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html)
- [MLxtend — Apriori Documentation](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Intro to Data Mining with Python](https://medium.com/@sujathamudadla1213/course-introduction-to-data-mining-in-python-beginner-module-data-preprocessing-b7087a67dc65)


*Submitted for IDAI105 | Data Mining Summative Assessment | InsightMart Analytics | BLACK FRIDAY // DATA CORE*
