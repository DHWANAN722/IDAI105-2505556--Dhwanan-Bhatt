# ⚡ BLACK FRIDAY // DATA CORE
### Mining the Future: Black Friday Sales Insights
**Data Mining – Year 1 Summative Assessment | CRS: Artificial Intelligence**
APP LINK: https://idai105-2505556--dhwanan-bhatt-zhgnhej6sxvysynfud5doz.streamlit.app/

---

## 🧠 Project Overview

**BLACK FRIDAY // DATA CORE** is an end-to-end data mining and business intelligence dashboard built for InsightMart Analytics. The project applies advanced data science techniques to a large-scale Black Friday retail transaction dataset, uncovering hidden patterns in customer shopping behavior, segmenting buyers into meaningful groups, discovering product association rules, and flagging anomalous transactions — all presented through a sleek, interactive cyberpunk-themed Streamlit dashboard deployed on the cloud.

The core mission: transform raw transactional data into actionable business intelligence that retail decision-makers can actually use. Who spends the most? Which products get bought together? Are there unusual high-spenders hiding in the data? This system answers all of it.

---

## 📦 Dataset

The dataset used is the **Black Friday Sales dataset**, containing over 550,000 retail transactions from a large retail chain during a Black Friday mega-sale event.

**Key columns:**
- `User_ID` — Unique customer identifier
- `Product_ID` — Unique product identifier
- `Gender` — Customer gender (M/F)
- `Age` — Age group (0–17, 18–25, 26–35, 36–45, 46–50, 51–55, 55+)
- `Occupation` — Occupation code (0–20)
- `City_Category` — City tier (A, B, C)
- `Stay_In_Current_City_Years` — Years living in current city
- `Marital_Status` — 0 = Single, 1 = Married
- `Product_Category_1/2/3` — Product category codes
- `Purchase` — Purchase amount in USD

---

## ⚙️ Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| Dashboard | Streamlit 1.55 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (KMeans, PCA, StandardScaler) |
| Association Mining | MLxtend (Apriori, TransactionEncoder) |
| Statistical Analysis | SciPy (Z-score, IQR) |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit Cloud |

---

## 🔬 Methodology — 8 Stage Pipeline

### Stage 1 · Project Scope Definition
The project begins by clearly defining objectives before any code is written. The four core goals established are: understanding shopping preferences across demographics, segmenting customers into behavioral clusters, identifying cross-selling opportunities through product associations, and detecting anomalous high-spend transactions. This scoping stage ensures every subsequent analysis step serves a concrete business purpose.

### Stage 2 · Data Cleaning & Preprocessing
Raw data is never analysis-ready. This stage handles all preparation tasks: missing values in `Product_Category_2` and `Product_Category_3` are filled with `0` (indicating no secondary purchase), duplicate rows are dropped, and categorical variables are encoded numerically. Gender becomes binary (M=0, F=1), age groups are mapped to ordered integers (1–7), city categories are encoded (A=1, B=2, C=3), and the "4+" stay year value is standardized to 4. Purchase amounts are normalized using `StandardScaler` to bring all features to the same scale for machine learning.

### Stage 3 · Exploratory Data Analysis (EDA)
The **📡 EDA SCAN** tab provides a comprehensive visual scan of the dataset. Key visualizations include average purchase amount broken down by age cohort and gender, top product category frequency charts, average revenue per category, and a full feature correlation heatmap. These charts reveal the foundational patterns in the data — which demographics dominate spending, which product categories generate the most revenue, and how features relate to each other.

### Stage 4 · Clustering Analysis
The **🧬 CLUSTER MAP** tab implements K-Means clustering on aggregated user-level features. The Elbow Method is used to determine the optimal number of clusters (k=4), and the resulting segments are labeled based on average spend: **Budget Shoppers**, **Casual Buyers**, **Regular Spenders**, and **Premium Buyers**. A PCA scatter plot projects the high-dimensional cluster space into 2D for visual inspection, and a statistics table summarizes the average spend, total spend, and transaction count for each segment.

### Stage 5 · Association Rule Mining
The **🔮 RULE ENGINE** tab runs the **Apriori algorithm** on user transaction baskets — each user's complete set of purchased product categories forms one transaction. Frequent itemsets are extracted and association rules are generated with configurable support and lift thresholds via interactive sliders. Rules are displayed ranked by lift score, showing which product category combinations appear together more often than chance. Example insight: *"Users who buy Category 1 products frequently also purchase Category 5 items"* — directly actionable for combo deal design.

### Stage 6 · Anomaly Detection
The **☢️ ANOMALY TRACE** tab flags statistically unusual transactions using the **Z-score method** (threshold adjustable via slider, default = 3). Transactions with a Z-score above the threshold are marked as anomalies and visualized in a scatter plot with glowing pink markers against the normal transaction cloud. An IQR boundary chart provides a secondary view of the spend distribution. Anomalous transactions are profiled by age cohort and listed in a detailed table for review.

### Stage 7 · Insights & Reporting
The **💀 INTEL REPORT** tab consolidates all findings into a mission debrief format. Key statistics — top spending age group, gender spend differential, highest-value city, most popular product category, anomaly rate, and cluster distribution — are displayed in styled neon stat cards. Six strategic recommendations are presented in a terminal-style readout, giving retail decision-makers direct, actionable intelligence from the analysis.

### Stage 8 · Deployment on Streamlit Cloud
The entire application is deployed as a live interactive web app on **Streamlit Cloud**, accessible via browser with no installation required. Users upload their own `train.csv` via the sidebar to initialize the analysis pipeline. All computations are cached for performance using `@st.cache_data`.

---

## 🎨 Design Philosophy

The dashboard uses a **cyberpunk neo-noir aesthetic** — deep void-black backgrounds, a cyan/pink/yellow neon color palette, Orbitron and Share Tech Mono typefaces, and glowing chart elements. Every component is styled to feel like a live intelligence terminal rather than a generic data dashboard. The UI is fully responsive and organized into five clearly labeled tabs for intuitive navigation.

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/YourUsername/IDAI105-YourID-YourName
cd IDAI105-YourID-YourName
pip install -r requirements.txt
streamlit run app.py
```
Upload `train.csv` via the sidebar when the app opens.

---

## 📁 Repository Structure

```
├── app.py                  # Main Streamlit dashboard (cyberpunk UI)
├── analysis.ipynb          # Full Jupyter Notebook — all 8 stages
├── requirements.txt        # Python dependencies
├── train.csv               # Black Friday dataset
└── README.md               # This file
```

---

## 📚 References

- [K-Means Clustering — Neptune.ai](https://neptune.ai/blog/k-means-clustering)
- [Association Rule Mining — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-market-basket-analysis/)
- [Anomaly Detection — DataCamp](https://www.datacamp.com/courses/anomaly-detection-in-python)
- [Customer Segmentation with K-Means](https://medium.com/@maleeshadesil va21/a-beginners-guide-to-clustering-customer-segmentation-with-k-means-clustering-c4e35c527ef8)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [MLxtend — Apriori](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)

---

## 👤 Submission Details

| Field | Value |
|---|---|
| Student Name | *(Dhwanan Bhatt)* |
| CRS | Artificial Intelligence |
| School | *(Udgam School For Children)* |
| Live App | [⚡ Launch App](https://idai105-2505556--dhwanan-bhatt-avvvz9hkwksrdi73fa5umb.streamlit.app/) |

---

*Submitted for IDAI105 | Data Mining Summative Assessment | InsightMart Analytics*
