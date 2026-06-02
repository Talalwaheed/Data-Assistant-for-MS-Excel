import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# --- Page Configuration ---
st.set_page_config(page_title="AI Advance Data Assistant", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div.stButton > button:first-child { background-color: #007bff; color: white; width: 100%; border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AI Advance Data Assistant")
st.markdown("**Universal Data Science & Machine Learning Pipeline**")
st.markdown("*Developed by Muhammad Basil & Talal Bin Waheed | University of Haripur*")

# --- SPEED OPTIMIZED AI LOADER ---
@st.cache_resource
def load_ai_logic():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_cached_embeddings(_model, text_list):
    return _model.encode(text_list)

# --- EXPORT FUNCTION ---
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Data Center")
    uploaded_file = st.file_uploader("Upload Dataset (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        st.success("✅ Dataset Connected")
        if st.button("🔄 Reset Data"):
            st.session_state.df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.rerun()
            
        st.divider()
        st.header("💾 Export Center")
        if 'df' in st.session_state:
            csv_data = convert_df(st.session_state.df)
            st.download_button(
                label="📥 Download Processed Data",
                data=csv_data,
                file_name='processed_dataset.csv',
                mime='text/csv',
            )

# --- Main App Logic ---
if uploaded_file:
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    df = st.session_state.df
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(include=['object', 'category'])

    # Dashboard Metrics (Generalized)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", df.shape[0])
    m2.metric("Total Features", df.shape[1])
    m3.metric("Numeric Features", num_df.shape[1])
    m4.metric("Data Integrity", f"{(1 - df.isnull().sum().sum()/df.size)*100:.1f}%")

    st.write("### 🛠️ Multi-Domain Analysis Command Center")
    
    commands = [
        "Select a command...",
        "🔹 SECTION 1: SMART EXPLORATION",
        "1. Preview Data (Top 5 Rows)",
        "2. Statistical Summary (Describe)",
        "3. Column Information (Data Types)",
        "4. Calculate Means/Averages",
        "5. Calculate Totals (Sum)",
        "6. Show Min/Max Values",
        "🔹 SECTION 2: AUTO-CLEANING",
        "7. Handle Missing Values (Smart-Fill)",
        "8. Remove Duplicate Records",
        "9. Standardize Text (Title Case)",
        "10. Normalize Data (Z-Score Scaling)",
        "11. Detect Data Outliers",
        "🔹 SECTION 3: VISUAL INTELLIGENCE",
        "12. Correlation Heatmap",
        "13. Interactive Bar Chart",
        "14. Box Plot (Distribution)",
        "15. Pie Chart (Categories)",
        "16. Line Chart (Trends)",
        "🔹 SECTION 4: ADVANCED AI (NLP)",
        "17. AI Semantic Clustering (Text Patterns)",
        "18. Visualize AI Clusters (t-SNE)",
        "19. Automated Insight Narrative",
        "🔹 SECTION 5: PREDICTIVE ML",
        "20. Train Prediction Model (Random Forest)",
        "21. Feature Importance (Driver Analysis)"
    ]

    selected = st.selectbox("Choose a processing task:", commands)
    
    if st.button("Execute Action"):
        if "SECTION" in selected or selected == "Select a command...":
            st.warning("Please select a valid command.")
        else:
            with st.spinner("🤖 Processing..."):
                
                # --- SECTION 1 ---
                if "Preview" in selected:
                    st.dataframe(df.head())
                elif "Summary" in selected:
                    st.write(df.describe())
                elif "Information" in selected:
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                elif "Means" in selected:
                    st.write(num_df.mean())
                elif "Totals" in selected:
                    st.write(num_df.sum())
                elif "Min/Max" in selected:
                    st.write("Maximum Values:", num_df.max())
                    st.write("Minimum Values:", num_df.min())

                # --- SECTION 2 ---
                elif "Outliers" in selected:
                    if not num_df.empty:
                        z = np.abs((num_df - num_df.mean()) / num_df.std())
                        outliers = df[(z > 3).any(axis=1)]
                        st.write(f"Identified {len(outliers)} statistical outliers.")
                        st.dataframe(outliers.head())
                elif "Missing" in selected:
                    st.session_state.df = df.fillna(df.median(numeric_only=True))
                    st.success("✅ Null values resolved using Median Imputation.")
                elif "Duplicate" in selected:
                    st.session_state.df = df.drop_duplicates()
                    st.success(f"✅ Removed {len(df)-len(st.session_state.df)} duplicate entries.")
                elif "Standardize" in selected:
                    for col in cat_df.columns:
                        st.session_state.df[col] = st.session_state.df[col].astype(str).str.strip().str.title()
                    st.success("✅ Text features standardized.")
                elif "Normalize" in selected:
                    scaler = StandardScaler()
                    st.session_state.df[num_df.columns] = scaler.fit_transform(num_df)
                    st.success("✅ Features normalized to standard Gaussian distribution.")

                # --- SECTION 3 ---
                elif "Heatmap" in selected:
                    fig, ax = plt.subplots(figsize=(10,6))
                    sns.heatmap(num_df.iloc[:, :12].corr(), annot=True, cmap='viridis', ax=ax)
                    st.pyplot(fig)
                elif "Bar Chart" in selected:
                    top_cols = num_df.mean().sort_values(ascending=False).head(10)
                    st.bar_chart(top_cols)
                elif "Box Plot" in selected:
                    fig, ax = plt.subplots()
                    sns.boxplot(data=num_df.iloc[:, :5], ax=ax)
                    st.pyplot(fig)
                elif "Pie Chart" in selected:
                    col = cat_df.columns[0] if not cat_df.empty else None
                    if col:
                        fig, ax = plt.subplots()
                        df[col].value_counts().head(8).plot.pie(autopct='%1.1f%%', ax=ax)
                        st.pyplot(fig)
                elif "Line Chart" in selected:
                    st.line_chart(num_df.iloc[:50, :2])

                # --- SECTION 4 ---
                elif "Clustering" in selected or "t-SNE" in selected:
                    if not cat_df.empty:
                        model = load_ai_logic()
                        items = df[cat_df.columns[0]].astype(str).unique()[:40]
                        embs = get_cached_embeddings(model, items)
                        if "Clustering" in selected:
                            km = KMeans(n_clusters=min(4, len(items))).fit(embs)
                            st.write(pd.DataFrame({"Feature Value": items, "AI Cluster ID": km.labels_}))
                        else:
                            tsne = TSNE(n_components=2, perplexity=min(5, len(items)-1)).fit_transform(embs)
                            fig, ax = plt.subplots()
                            ax.scatter(tsne[:,0], tsne[:,1], alpha=0.7)
                            for i, txt in enumerate(items): ax.annotate(txt, (tsne[i,0], tsne[i,1]))
                            st.pyplot(fig)
                
                elif "Insights" in selected:
                    st.write("#### 🧠 Automated Insights Engine")
                    if not num_df.empty:
                        top_stat = num_df.mean().idxmax()
                        st.info(f"💡 Statistical Lead: The feature with the highest mean density is **{top_stat}**.")
                    if not cat_df.empty:
                        top_val = df[cat_df.columns[0]].mode()[0]
                        st.info(f"💡 Categorical Mode: The most dominant class in the dataset is **{top_val}**.")
                    st.success("Analysis complete. Dataset patterns successfully mapped.")

                # --- SECTION 5 ---
                elif "Train" in selected:
                    if num_df.shape[1] >= 2:
                        target = num_df.columns[-1]
                        X = num_df.drop(columns=[target]).fillna(0).iloc[:1500]
                        y = num_df[target].iloc[:1500]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
                        st.metric("Predictive Accuracy (R² Score)", f"{model.score(X_test, y_test):.2f}")
                
                elif "Importance" in selected:
                    if num_df.shape[1] > 1:
                        target = num_df.columns[-1]
                        corrs = num_df.corr()[target].abs().sort_values(ascending=False)[1:11]
                        st.bar_chart(corrs)

    # --- Accuracy Lab ---
    st.divider()
    st.write("### 🎯 Accuracy Builder Laboratory")
    if len(num_df.columns) >= 2:
        c1, c2 = st.columns(2)
        target_var = c1.selectbox("Target Variable:", num_df.columns, index=len(num_df.columns)-1)
        features = c2.multiselect("Input Features:", [c for c in num_df.columns if c != target_var])
        if st.button("🚀 Train Model") and features:
            X = num_df[features].fillna(0)
            y = num_df[target_var].fillna(0)
            X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2)
            rf = RandomForestRegressor().fit(X_t, y_t)
            st.metric("Test Accuracy", f"{rf.score(X_v, y_v):.4f}")
else:
    st.info("👋 System Standby: Please upload a dataset to initialize the pipeline.")