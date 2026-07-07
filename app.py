import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

# --- Helper Functions ---
def show_image(path, caption=None):
    path = Path(path)
    if path.exists():
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {path}")

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f4068, #162447, #e43f5a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        color: #495057;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2D3748;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)
# --- Constants ---
EDUCATION_MAP = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}

SEGMENT_COLORS = {
    "Premium Customers": "#1a508b", # Deep premium blue
    "Family Shoppers": "#d97736",    # Warm brown-orange
    "Budget Shoppers": "#0f766e"     # Deep teal
}

# --- Feature Engineering Helpers ---
def compute_spending_features(wines, fruits, meat, fish, sweets, gold):
    """Compute total spending and category spending shares."""
    total_spending = wines + fruits + meat + fish + sweets + gold
    if total_spending > 0:
        return (
            total_spending,
            wines / total_spending,
            fruits / total_spending,
            meat / total_spending,
            fish / total_spending,
            sweets / total_spending,
            gold / total_spending
        )
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def compute_purchase_features(web_purchases, catalog_purchases, store_purchases):
    """Compute total purchases and channel ratios."""
    total_purchases = web_purchases + catalog_purchases + store_purchases
    if total_purchases > 0:
        return (
            total_purchases,
            web_purchases / total_purchases,
            catalog_purchases / total_purchases,
            store_purchases / total_purchases
        )
    return 0.0, 0.0, 0.0, 0.0

def make_bar_chart(df, y_col, title, y_label):
    """Generate a standardized comparison bar chart for segment metrics."""
    fig = px.bar(
        df,
        x="Customer_Segment",
        y=y_col,
        title=title,
        labels={y_col: y_label, "Customer_Segment": "Customer Segment"},
        color="Customer_Segment",
        color_discrete_map=SEGMENT_COLORS,
        text_auto=".2s"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(t=45, b=0, l=0, r=0))
    return fig

# --- Load Model Artifacts ---
@st.cache_resource
def load_model():
    model_path = "customer_segmentation_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model artifacts: {e}")
    return None

model_artifacts = load_model()

# Stop execution immediately if model artifacts are not loaded
if model_artifacts is None:
    st.error("⚠️ Model artifacts file (`customer_segmentation_model.pkl`) not found in the workspace. Please train the model first.")
    st.stop()

# --- Build Persona Lookup ---
if "customer_personas" in model_artifacts:
    personas_lookup = (
        model_artifacts["customer_personas"]
        .set_index("Segment")
        .to_dict("index")
    )
else:
    personas_lookup = {}

# --- Segment Persona Details Helper ---
def get_persona_details(segment_name):
    """Return business profile and recommendations for a customer segment."""
    data = personas_lookup.get(segment_name)
    if data is None:
        return {
            "Profile": "Not Available",
            "Behavior": "Not Available",
            "Core Strategy": "Not Available",
            "Actionable Recommendations": [],
        }
    return {
        "Profile": data["Profile"],
        "Behavior": data["Behavior"],
        "Core Strategy": data["Core Strategy"],
        "Actionable Recommendations": [
            line.strip()
            for line in data["Actionable Recommendations"].split("\n")
            if line.strip()
        ],
    }

# --- Sidebar Navigation ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <h2 style="color: #1f4068; margin-bottom: 5px; font-weight: 800;">👥 Customer Segmentation</h2>
    <p style="color: #718096; font-size: 0.9rem;">Customer Segmentation Platform</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.divider()

menu = st.sidebar.radio(
    "Navigation",
    ["Predict Customer Segment", "Segment Explorer", "Segment Analytics", "About Project"]
)

# Project quick facts in sidebar
st.sidebar.divider()
st.sidebar.subheader("Model Status")
st.sidebar.success("✅ Ready")
st.sidebar.markdown(f"""
**Algorithm**  
K-Means (K={len(model_artifacts['segment_map'])})

**Customer Segments**  
{len(model_artifacts['segment_map'])}

**Silhouette Score**  
0.159
""")

# --- Page Logic ---
if menu == "Predict Customer Segment":
    st.markdown('<div class="main-title">Customer Segmentation Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter customer characteristics below to classify them and view tailor-made marketing strategies.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Customer Information Form</div>', unsafe_allow_html=True)
    
    # Extract training features scaler and means for default fallback values
    scaler = model_artifacts["scaler"]
    feature_names = model_artifacts["feature_names"]
    feature_means = dict(zip(feature_names, scaler.mean_))
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics")
            age = st.slider("Age", min_value=18, max_value=100, value=45, help="Customer's age in years")
            education = st.selectbox("Education Level", ["Basic", "2n Cycle", "Graduation", "Master", "PhD"], index=2, help="Highest education level achieved")
            income = st.number_input("Annual Income (₹)", min_value=0, max_value=250000, value=55000, step=1000, help="Annual household income")
            family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3, help="Total number of members in the family")
            children = st.number_input("Children in Family", min_value=0, max_value=10, value=1, help="Number of children/teenagers in the household")
            tenure = st.number_input("Customer Tenure (Days)", min_value=0, max_value=2000, value=365, help="Number of days the customer has been with the company")
            recency = st.slider("Recency (Days)", min_value=0, max_value=100, value=45, help="Number of days since customer's last purchase")

        with col2:
            st.markdown("#### 🛒 Last 2 Years Spending (₹)")
            wines = st.number_input("Wines spending", min_value=0, value=250, step=10, help="Spending on wine")
            meat = st.number_input("Meat spending", min_value=0, value=150, step=10, help="Spending on meat products")
            fruits = st.number_input("Fruits spending", min_value=0, value=30, step=5, help="Spending on fruits")
            fish = st.number_input("Fish spending", min_value=0, value=40, step=5, help="Spending on fish products")
            sweets = st.number_input("Sweets spending", min_value=0, value=30, step=5, help="Spending on sweet products")
            gold = st.number_input("Gold products spending", min_value=0, value=50, step=5, help="Spending on gold products")
            
        with col3:
            st.markdown("#### 🎯 Purchases & Channels")
            web_purchases = st.number_input("Web Purchases", min_value=0, value=4, step=1, help="Purchases via website")
            catalog_purchases = st.number_input("Catalog Purchases", min_value=0, value=2, step=1, help="Purchases via catalogs")
            store_purchases = st.number_input("Store Purchases", min_value=0, value=5, step=1, help="Purchases in physical stores")
            campaign_response = st.slider("Accepted Campaigns", min_value=0, max_value=6, value=1, help="Number of marketing campaigns accepted by the customer")
            
            st.markdown("---")
            st.caption("ℹ️ Total Spending, Total Purchases, Spending Shares, and Purchase Ratios are automatically computed to ensure logical consistency.")

        # Collapsible Advanced Features
        with st.expander("⚙️ Advanced Model Features"):
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                deals_purchases = st.number_input("Deals Purchases", min_value=0, value=2, help="Number of purchases made with discount/deals")
            with col_a2:
                web_visits = st.number_input("Web Visits / Month", min_value=0, value=5, help="Number of visits to the company website last month")
            with col_a3:
                complain = st.selectbox("Customer Complained?", ["No", "Yes"], index=0, help="Has customer complained in the last 2 years?")
                complain_val = 1 if complain == "Yes" else 0
                
        submit_button = st.form_submit_button(label="Predict Customer Segment", type="primary")

    if submit_button:
        # 1. Map Education
        education_val = EDUCATION_MAP[education]
        
        # 2. Compute spending features
        total_spending, wines_share, fruits_share, meat_share, fish_share, sweets_share, gold_share = compute_spending_features(
            wines, fruits, meat, fish, sweets, gold
        )
        
        # 3. Compute purchase features
        total_purchases, web_ratio, catalog_ratio, store_ratio = compute_purchase_features(
            web_purchases, catalog_purchases, store_purchases
        )
        
        # Validation alerts
        if total_spending == 0:
            st.warning("ℹ️ Total spending is zero. Product preferences will all be treated as zero.")
        if total_purchases == 0:
            st.warning("ℹ️ Total purchases are zero. Purchase ratios will all be treated as zero.")
        
        # 4. Construct input dictionary
        input_dict = {
            "Education": education_val,
            "Income": income,
            "Recency": recency,
            "NumDealsPurchases": deals_purchases,
            "NumWebVisitsMonth": web_visits,
            "Complain": complain_val,
            "Tenure": tenure,
            "Age": age,
            "Children": children,
            "Family_Size": family_size,
            "Total_Spending": total_spending,
            "Total_Purchases": total_purchases,
            "Web_Purchase_Ratio": web_ratio,
            "Store_Purchase_Ratio": store_ratio,
            "Catalog_Purchase_Ratio": catalog_ratio,
            "Campaign_Acceptance_Count": campaign_response,
            "Wines_Share": wines_share,
            "Fruits_Share": fruits_share,
            "MeatProducts_Share": meat_share,
            "FishProducts_Share": fish_share,
            "SweetProducts_Share": sweets_share,
            "GoldProds_Share": gold_share
        }
        
        df_input = pd.DataFrame([input_dict])
        
        # Reorder columns to match feature_names exactly
        df_input = df_input[model_artifacts["feature_names"]]
        
        # Scale & Predict
        scaled_input = model_artifacts["scaler"].transform(df_input)
        cluster = model_artifacts["kmeans"].predict(scaled_input)[0]
        predicted_segment = model_artifacts["segment_map"][cluster]
        
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        # Display beautiful segment card
        segment_colors = {
            "Premium Customers": "#1a508b", # Deep premium blue
            "Family Shoppers": "#d97736",    # Warm brown-orange
            "Budget Shoppers": "#0f766e"     # Deep teal
        }
        seg_color = segment_colors.get(predicted_segment, "#2B6CB0")
        
        st.markdown(f"""
        <div style="background-color: {seg_color}; padding: 25px; border-radius: 12px; text-align: center; color: white; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: white;">⭐ {predicted_segment} ⭐</h1>
            <p style="margin: 5px 0 0 0; font-size: 1.1rem; opacity: 0.9;">Customer Segment Classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dynamic prediction visualization tabs
        res_tab1, res_tab2, res_tab3 = st.tabs([
            "🔍 Segment Insights",
            "⚖️ Segment Comparison",
            "📊 Purchase Breakdown"
        ])
        
        with res_tab1:
            details = get_persona_details(predicted_segment)
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("### 📋 Segment Persona")
                st.markdown(f"**Target Profile:** {details['Profile']}")
                st.markdown(f"**Core Behavior:** {details['Behavior']}")
                st.markdown(f"**Marketing Direction:** `{details['Core Strategy']}`")
            
            with col_s2:
                st.markdown("### 💡 Recommended Marketing Actions")
                for rec in details['Actionable Recommendations']:
                    if rec.strip():
                        st.markdown(rec)
                        
        with res_tab2:
            cluster_profile = model_artifacts["cluster_profile"]
            if predicted_segment in cluster_profile.index:
                st.markdown("### ⚖️ Customer Profile vs Segment Averages")
                
                # Metrics configuration
                metrics_to_compare = {
                    "Income": (income, "Income", "₹{:,.2f}"),
                    "Spending": (total_spending, "Total_Spending", "₹{:,.2f}"),
                    "Purchases": (total_purchases, "Total_Purchases", "{:.2f}"),
                    "Age": (age, "Age", "{:.1f} yrs"),
                    "Family Size": (family_size, "Family_Size", "{:.2f}"),
                    "Recency": (recency, "Recency", "{:.1f} days"),
                    "Campaign Acceptance": (campaign_response, "Campaign_Acceptance_Count", "{:.2f}")
                }
                
                comp_rows = []
                for label, (cust_val, col_name, fmt) in metrics_to_compare.items():
                    avg_val = cluster_profile.loc[predicted_segment, col_name]
                    row = {
                        "Metric": label,
                        "Customer": fmt.format(cust_val),
                        "Segment Average": fmt.format(avg_val)
                    }
                    comp_rows.append(row)
                    
                comp_df = pd.DataFrame(comp_rows)
                st.table(comp_df.set_index("Metric"))
                
                # Summary Bullet Points
                st.markdown("---")
                avg_inc = cluster_profile.loc[predicted_segment, "Income"]
                avg_spend = cluster_profile.loc[predicted_segment, "Total_Spending"]
                avg_purch = cluster_profile.loc[predicted_segment, "Total_Purchases"]
                
                def get_comparison_phrase(cust_val, avg_val):
                    diff_pct = (cust_val - avg_val) / avg_val if avg_val > 0 else 0
                    if abs(diff_pct) <= 0.05:
                        return "On average"
                    elif diff_pct > 0.05:
                        return "Slightly above average" if diff_pct <= 0.2 else "Above average"
                    else:
                        return "Slightly below average" if diff_pct >= -0.2 else "Below average"
                
                inc_phrase = get_comparison_phrase(income, avg_inc)
                spend_phrase = get_comparison_phrase(total_spending, avg_spend)
                purch_phrase = get_comparison_phrase(total_purchases, avg_purch)
                
                segment_singular = predicted_segment.replace("Customers", "Customer").replace("Shoppers", "Shopper")
                
                st.markdown(f"**Compared with the average {segment_singular}:**")
                st.markdown(f"• **Income**: {inc_phrase}")
                st.markdown(f"• **Spending**: {spend_phrase}")
                st.markdown(f"• **Purchase Frequency**: {purch_phrase}")
            else:
                st.warning("Cluster profile data not available for comparison.")
                
        with res_tab3:
            st.markdown("### 📊 Customer Profile Breakdown")
            col_ch1, col_ch2 = st.columns(2)
            
            with col_ch1:
                spend_labels = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"]
                spend_vals = [wines, fruits, meat, fish, sweets, gold]
                if sum(spend_vals) > 0:
                    fig_spend = px.pie(
                        names=spend_labels,
                        values=spend_vals,
                        title="Product Category Spending Breakdown",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_spend.update_layout(showlegend=True, margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_spend, use_container_width=True)
                    st.metric(
                        "Total Spending",
                        f"₹{total_spending:,.2f}"
                    )
                else:
                    st.info("No spending recorded to visualize.")
                    
            with col_ch2:
                channel_labels = ["Web Purchases", "Catalog Purchases", "Store Purchases"]
                channel_vals = [web_purchases, catalog_purchases, store_purchases]
                if sum(channel_vals) > 0:
                    fig_chan = px.pie(
                        names=channel_labels,
                        values=channel_vals,
                        title="Purchase Channel Distribution",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    fig_chan.update_layout(showlegend=True, margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_chan, use_container_width=True)
                    st.metric(
                        "Total Purchases",
                        total_purchases
                    )
                else:
                    st.info("No purchases recorded to visualize.")

elif menu == "Segment Explorer":
    st.markdown('<div class="main-title">Segment Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explore the analytical visualizations generated during model training.</div>', unsafe_allow_html=True)
    
    tab_comp, tab_pca, tab_heatmap = st.tabs([
        "📦 Feature Comparison",
        "🗺️ PCA Visualization",
        "🔥 Segment Heatmap"
    ])
    
    with tab_comp:
        st.markdown("### 📦 Customer Segment Comparison Boxplots")
        st.write("This boxplot visualization shows the distribution of key features across the three detected customer segments.")
        show_image("images/customer_segment_comparison.png")
        
        st.markdown("""
        **Key Insights**
        * **Income & Spending separation**: Clear visual boundaries show Premium Customers as the high-value cluster, while Budget Shoppers cluster at the lower end.
        * **Household dynamics**: Clear differentiation between the single/couples-dominated Premium Customers and the child-heavy Family Shoppers.
        """)
        
    with tab_pca:
        st.markdown("### 🗺️ Customer Segments in PCA Space")
        st.write("This plot displays customer segments projected on the first two Principal Components (PCA). Centroids of clusters are marked with an 'X'.")
        show_image("images/customer_segments_pca.png")
        st.info("The first two principal components capture the majority of the dataset's variance while providing an intuitive 2D visualization of customer segments.")
        
        st.markdown("""
        **Key Insights**
        * **Premium Customers**: Form a clearly distinguishable segment.
        * **Family & Budget Shoppers**: Exhibit moderate overlap, indicating some shared purchasing characteristics.
        """)
        
    with tab_heatmap:
        st.markdown("### 🔥 Customer Segment Profile Heatmap")
        st.write("This heatmap visualizes the average features for each cluster.")
        show_image("images/customer_segment_profile_heatmap.png")
        st.info("Colors represent normalized feature values, while the annotations display the original cluster means.")
        
        st.markdown("""
        **Key Insights**
        * **Premium Customers**: Have the highest income, spending, and campaign engagement.
        * **Family Shoppers**: Are characterized by larger households.
        * **Budget Shoppers**: Are younger and spend less overall.
        """)

elif menu == "Segment Analytics":
    st.markdown('<div class="main-title">Segment Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Visualizing the key characteristics and metrics of the learned customer segments.</div>', unsafe_allow_html=True)
    
    feature_names = model_artifacts["feature_names"]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customer Segments", len(model_artifacts['segment_map']))
    col2.metric("Features Modelled", len(feature_names))
    col3.metric("Algorithm", "K-Means")
    col4.metric("Silhouette Score", "0.159")
    
    st.markdown('<div class="section-header">Segment Metrics Comparison</div>', unsafe_allow_html=True)
    
    # Reset index to make Customer_Segment a column for plotting
    profile_plot_df = model_artifacts["cluster_profile"].reset_index()
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_inc = make_bar_chart(profile_plot_df, "Income", "Average Annual Income by Segment", "Average Income (₹)")
        st.plotly_chart(fig_inc, use_container_width=True)
        
    with col_chart2:
        fig_spend = make_bar_chart(profile_plot_df, "Total_Spending", "Average Total Spending by Segment", "Average Spending (₹)")
        st.plotly_chart(fig_spend, use_container_width=True)
        
    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        fig_purch = make_bar_chart(profile_plot_df, "Total_Purchases", "Average Total Purchases by Segment", "Average Purchase Count")
        st.plotly_chart(fig_purch, use_container_width=True)
        
    with col_chart4:
        CLUSTER_SIZES = {
            "Family Shoppers": 924,
            "Premium Customers": 676,
            "Budget Shoppers": 612
        }
        total_size = sum(CLUSTER_SIZES.values())
        size_data = [
            {"Customer_Segment": seg, "Percentage": count / total_size * 100}
            for seg, count in CLUSTER_SIZES.items()
        ]
        df_sizes = pd.DataFrame(size_data)
        
        fig_size = make_bar_chart(df_sizes, "Percentage", "Segment Size Distribution (%)", "Percentage of Customer Base (%)")
        st.plotly_chart(fig_size, use_container_width=True)
    st.markdown('<div class="section-header">Key Customer Segment Summary</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(profile_plot_df))
    style_map = {
        "Premium Customers": st.info,
        "Family Shoppers": st.warning,
        "Budget Shoppers": st.success
    }
    
    for i, seg in enumerate(profile_plot_df["Customer_Segment"]):
        row = model_artifacts["cluster_profile"].loc[seg]
        card_func = style_map.get(seg, st.write)
        
        summary_text = f"""
        **{seg}**
        
        • **Income**: ₹{row['Income']:,.0f}  
        • **Spending**: ₹{row['Total_Spending']:,.0f}  
        • **Purchases**: {row['Total_Purchases']:.1f}  
        • **Family Size**: {row['Family_Size']:.1f}
        """
        with cols[i]:
            card_func(summary_text)

elif menu == "About Project":
    st.markdown('<div class="main-title">About Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An End-to-End Customer Segmentation Project for Personalized Marketing</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📌 Project Overview
    This project segments customers of a retail business based on their demographics, purchasing behaviors, and interactions with promotional campaigns. 
    By applying **unsupervised clustering**, the business can transition from generic mass-marketing to highly personalized campaigns, resulting in higher conversion rates, optimized marketing spend, and improved customer lifetime value (LTV).
    
    The final model was exported using Joblib and deployed through an interactive Streamlit dashboard capable of segmenting new customers and providing business recommendations.

    ### ⚙️ Project Workflow
    The unsupervised machine learning workflow implemented in this project consists of the following steps:
    1. **Data Cleaning & Preprocessing**: Handled missing values (e.g., in Income), parsed date features (`Dt_Customer`), and removed outliers using statistical IQR methods.
    2. **Feature Engineering**:
       - **Demographics**: Computed customer `Age` from birth year, customer `Tenure` from enrollment date, and household `Family_Size` combining marital status and children counts.
       - **Spending Profiles**: Consolidated spending across 6 categories (Wines, Fruits, Meat, Fish, Sweets, Gold) and computed spending shares.
       - **Channel Preferences**: Formulated web, catalog, and store purchase ratios.
       - **Marketing Engagement**: Summed accepted campaign counts to gauge campaign responsiveness.
    3. **Feature Scaling**: Normalized the engineered features using a `StandardScaler` to ensure the distance metrics are not biased by feature scales.
    4. **Optimal Cluster Selection**: Evaluated the Elbow Method, Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index, and business interpretability to select **K = 3**.
    5. **Clustering Algorithm**: Applied **K-Means Clustering** with `k-means++` initialization to segment the customer base.
    6. **Dimensionality Reduction**: Visualized the clusters in 2D space using **Principal Component Analysis (PCA)**.

    ### 💎 Key Customer Segments Identified
    - **Premium Customers (Cluster 1)**: High-income, low household size, extremely high spending across premium products (especially Wines and Meat), and highly responsive to campaigns.
    - **Family Shoppers (Cluster 0)**: Moderate income, larger households, and consistent purchasing behavior.
    - **Budget Shoppers (Cluster 2)**: Lower income, younger customers with lower overall spending and purchase frequency.
    
    ### 🏆 Project Highlights
    - **22 engineered customer features**
    - **3 distinct customer segments identified**
    - **K-Means selected** after comparison with Agglomerative Clustering
    - **Interactive Streamlit dashboard** deployed for customer segmentation and business recommendation delivery
    """)
    
    st.divider()
    st.markdown("### 🛠️ Tech Stack Used")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("**Core Libraries**\n- Python 3\n- Pandas\n- NumPy\n- Joblib")
    with col_t2:
        st.markdown("**Machine Learning**\n- Scikit-Learn\n- SciPy (Hierarchical Clustering)")
    with col_t3:
        st.markdown("**Frontend & Visualizations**\n- Streamlit Dashboard\n- Plotly Express\n- Seaborn & Matplotlib")
