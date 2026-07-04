import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #C0392B;
    }
    .metric-label { font-size: 0.78rem; color: #7F8C8D; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.7rem; font-weight: 700; color: #2C3E50; }
    h1 { color: #2C3E50 !important; }
    h2, h3 { color: #2C3E50 !important; }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#C0392B', '#2C3E50', '#E67E22', '#27AE60', '#2980B9', '#8E44AD']

# ── Load data & train model ───────────────────────────────────
@st.cache_resource
def load_all():
    rfm = pd.read_csv('rfm_clean.csv')
    tx  = pd.read_csv('transactions_clean.csv', parse_dates=['InvoiceDate'])

    # Train model fresh (avoids pickle version mismatch)
    num_features = ['Frequency', 'Monetary', 'AvgOrderValue']
    cat_features = ['Country_grp']
    X = rfm[num_features + cat_features]
    y = rfm['Churn']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    return rfm, tx, model

rfm, tx, model = load_all()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/240px-Octicons-mark-github.svg.png",
    width=40
)
st.sidebar.title("🔧 Dashboard Controls")
st.sidebar.markdown("---")

# Interactive feature 1 — Country filter
all_countries = ['All Countries'] + sorted(rfm['Country'].unique().tolist())
selected_country = st.sidebar.selectbox(
    "🌍 Filter by Country",
    options=all_countries,
    index=0
)

# Interactive feature 2 — Recency slider
max_recency = int(rfm['Recency'].max())
recency_range = st.sidebar.slider(
    "📅 Recency Range (days since last purchase)",
    min_value=0,
    max_value=max_recency,
    value=(0, max_recency),
    step=10
)

# Interactive feature 3 — Churn filter checkbox
show_churned_only = st.sidebar.checkbox("Show churned customers only", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**MRTB 2173 — Agile Data Science**")
st.sidebar.markdown("Customer Churn Analytics Dashboard")

# ── Filter data ───────────────────────────────────────────────
filtered = rfm.copy()
if selected_country != 'All Countries':
    filtered = filtered[filtered['Country'] == selected_country]
filtered = filtered[
    (filtered['Recency'] >= recency_range[0]) &
    (filtered['Recency'] <= recency_range[1])
]
if show_churned_only:
    filtered = filtered[filtered['Churn'] == 1]

# ── Header ────────────────────────────────────────────────────
st.title("📊 Customer Churn Analytics Dashboard")
st.markdown(
    f"Showing **{len(filtered):,}** customers"
    + (f" in **{selected_country}**" if selected_country != 'All Countries' else " across **all countries**")
    + f" | Recency: **{recency_range[0]}–{recency_range[1]} days**"
)
st.markdown("---")

# ── KPI metrics ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

churn_rate = filtered['Churn'].mean() * 100 if len(filtered) > 0 else 0
avg_recency = filtered['Recency'].mean() if len(filtered) > 0 else 0
avg_monetary = filtered['Monetary'].mean() if len(filtered) > 0 else 0
total_customers = len(filtered)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Customers</div>
        <div class="metric-value">{total_customers:,}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Churn Rate</div>
        <div class="metric-value">{churn_rate:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Recency (days)</div>
        <div class="metric-value">{avg_recency:.0f}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Spend (£)</div>
        <div class="metric-value">£{avg_monetary:,.0f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# VISUALISATION 1 — Churn vs Active (Pie)
# VISUALISATION 2 — Top 10 Countries by Customer Count (Bar)
# ═══════════════════════════════════════════════════════════════
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📌 Churn vs Active Customers")
    churn_counts = rfm['Churn'].value_counts()
    labels = ['Active', 'Churned']
    sizes  = [churn_counts.get(0, 0), churn_counts.get(1, 0)]
    colors = ['#27AE60', '#C0392B']

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight('bold')
        t.set_color('white')
    ax1.set_title('Customer Churn Distribution', fontweight='bold', color='#2C3E50', pad=12)
    fig1.patch.set_alpha(0)
    st.pyplot(fig1)
    plt.close()

with col_b:
    st.subheader("🌍 Top 10 Countries by Customers")
    top_c = rfm.groupby('Country').size().sort_values(ascending=False).head(10)
    colors_bar = [PALETTE[0]] + [PALETTE[1]] * 9

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars = ax2.barh(top_c.index[::-1], top_c.values[::-1],
                    color=colors_bar[::-1], edgecolor='white')
    ax2.set_xlabel('Number of Customers')
    ax2.set_title('Top 10 Countries', fontweight='bold', color='#2C3E50')
    for bar in bars:
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f'{int(bar.get_width()):,}', va='center', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig2.patch.set_alpha(0)
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# VISUALISATION 3 — RFM Scatter (Recency vs Monetary)
# ═══════════════════════════════════════════════════════════════
st.subheader("💰 RFM Analysis — Recency vs Monetary Value")

col_c, col_d = st.columns([2, 1])

with col_c:
    plot_df = filtered.sample(min(1000, len(filtered)), random_state=42) if len(filtered) > 0 else filtered
    churn_colors = plot_df['Churn'].map({0: '#27AE60', 1: '#C0392B'})

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    scatter = ax3.scatter(
        plot_df['Recency'],
        plot_df['Monetary'].clip(upper=5000),
        c=churn_colors, alpha=0.5, s=18, edgecolors='none'
    )
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27AE60', label='Active'),
        Patch(facecolor='#C0392B', label='Churned')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')
    ax3.set_xlabel('Recency (days since last purchase)')
    ax3.set_ylabel('Monetary Value (£, capped at £5,000)')
    ax3.set_title('Recency vs Monetary Value by Churn Status', fontweight='bold', color='#2C3E50')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    fig3.patch.set_alpha(0)
    st.pyplot(fig3)
    plt.close()

with col_d:
    st.markdown("**Key Insight**")
    st.info(
        "Churned customers (red) cluster toward higher Recency values "
        "(they haven't purchased in a long time). Active customers (green) "
        "tend to have lower Recency and higher Monetary value. "
        "This confirms Recency is the strongest churn signal."
    )
    if len(filtered) > 0:
        churned_df  = filtered[filtered['Churn'] == 1]
        active_df   = filtered[filtered['Churn'] == 0]
        st.metric("Avg Recency (Churned)",  f"{churned_df['Recency'].mean():.0f} days")
        st.metric("Avg Recency (Active)",   f"{active_df['Recency'].mean():.0f} days")
        st.metric("Avg Spend (Churned)",    f"£{churned_df['Monetary'].mean():,.0f}")
        st.metric("Avg Spend (Active)",     f"£{active_df['Monetary'].mean():,.0f}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# PREDICTIVE OUTPUT — Churn Probability Calculator
# ═══════════════════════════════════════════════════════════════
st.subheader("🤖 Churn Probability Predictor")
st.markdown("Enter a customer's RFM values below to get an instant churn risk prediction from the Random Forest model.")

pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)

with pred_col1:
    input_freq = st.number_input(
        "Frequency (orders)", min_value=1, max_value=200, value=5, step=1
    )
with pred_col2:
    input_monetary = st.number_input(
        "Monetary Value (£)", min_value=1.0, max_value=50000.0, value=500.0, step=50.0
    )
with pred_col3:
    input_avg = st.number_input(
        "Avg Order Value (£)", min_value=1.0, max_value=5000.0, value=100.0, step=10.0
    )
with pred_col4:
    input_country = st.selectbox(
        "Country",
        options=['United Kingdom', 'Germany', 'France', 'Australia', 'Netherlands', 'Other'],
        index=0
    )

if st.button("🔮 Predict Churn Risk", type="primary"):
    input_df = pd.DataFrame({
        'Frequency':     [input_freq],
        'Monetary':      [input_monetary],
        'AvgOrderValue': [input_avg],
        'Country_grp':   [input_country]
    })

    proba = model.predict_proba(input_df)[0][1]
    pred  = model.predict(input_df)[0]

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        color = "#C0392B" if pred == 1 else "#27AE60"
        label = "⚠️ HIGH CHURN RISK" if pred == 1 else "✅ LOW CHURN RISK"
        st.markdown(f"""
        <div style="background:{color};padding:1.2rem;border-radius:10px;text-align:center">
            <span style="color:white;font-size:1.2rem;font-weight:700">{label}</span>
        </div>""", unsafe_allow_html=True)

    with res_col2:
        st.metric("Churn Probability", f"{proba*100:.1f}%")

    with res_col3:
        if pred == 1:
            st.warning("**Recommendation:** This customer is at high risk. Consider a targeted retention campaign — discount voucher or personalised outreach.")
        else:
            st.success("**Recommendation:** This customer appears active. Monitor Recency over the next 30 days to detect early churn signals.")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#95A5A6;font-size:0.8rem'>"
    "MRTB 2173 Agile Data Science — PMA Dashboard | Beverly | MSc Business Intelligence & Analytics"
    "</div>",
    unsafe_allow_html=True
)
