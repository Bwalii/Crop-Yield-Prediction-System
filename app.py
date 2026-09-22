import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

# =====================================================
# 1. PAGE CONFIG & CUSTOM STYLING
# =====================================================
st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #f5f9f6; }
    .main-title { font-size: 38px; font-weight: bold; color: #145A32; text-align: center; }
    .sub-title { text-align: center; color: #555555; font-size: 16px; margin-bottom: 25px; }
    .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); }
    .stButton>button { width: 100%; height: 48px; border-radius: 8px; font-size: 16px; font-weight: bold; background: #1E8449; color: white; }
    .stButton>button:hover { background: #145A32; color: white; }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

USERNAME, PASSWORD = "admin", "crop123"

# =====================================================
# 2. FILE LOADING & HELPER FUNCTIONS
# =====================================================
@st.cache_resource
def load_assets():
    try:
        model = XGBRegressor()
        model.load_model("xgb_crop_yield_model.json")

        with open("le_state.pkl", "rb") as f:
            le_state = pickle.load(f)
        with open("le_crop.pkl", "rb") as f:
            le_crop = pickle.load(f)
        with open("le_season.pkl", "rb") as f:
            le_season = pickle.load(f)

        rainfall = pd.read_csv("rainfall_cleaned.csv")
        rainfall.columns = rainfall.columns.str.lower().str.strip()

        return model, le_state, le_crop, le_season, rainfall
    except Exception as e:
        st.error(f"Error loading required files: {e}")
        st.stop()


def estimate_rainfall(rainfall_df, state, year):
    data = rainfall_df[rainfall_df["state"] == state].dropna(subset=["rainfall"])

    if data.empty:
        return float(rainfall_df["rainfall"].mean())

    if year in data["year"].values:
        return float(data.loc[data["year"] == year, "rainfall"].iloc[0])

    # Polynomial extrapolation for missing/future years
    X = data[["year"]]
    y = data["rainfall"]
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    reg = LinearRegression()
    reg.fit(X_poly, y)

    pred = reg.predict(poly.transform([[year]]))[0]
    return float(max(pred, 0))


# =====================================================
# 3. AUTHENTICATION (LOGIN)
# =====================================================
def login_screen():
    st.markdown("<h1 class='main-title'>🌾 Crop Yield Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Machine Learning Based Agricultural Decision Support System</p>", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔐 Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if user == USERNAME and pwd == PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Username or Password")
        st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state.logged_in:
    login_screen()
    st.stop()

# Load dataset and artifacts
model, le_state, le_crop, le_season, rainfall = load_assets()

# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================
st.sidebar.image("https://img.icons8.com/color/96/wheat.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["🏠 Home", "🌾 Prediction", "📊 Dashboard", "👤 About"])

st.sidebar.write("")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =====================================================
# 5. PAGE ROUTING
# =====================================================

# --- HOME PAGE ---
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>🌾 Crop Yield Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Machine Learning Based Agricultural Decision Support System</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Algorithm", "XGBoost Regressor")
    col2.metric("Target Variable", "Crop Yield (tons/ha)")

    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("System Overview")
        st.write("""
        This platform leverages XGBoost machine learning to predict crop yields using historical cultivation and rainfall data.
        
        * **Inputs:** State, Crop Type, Season, Year, Area (Hectares), and Rainfall (mm).
        * **Rainfall Forecasting:** Automated trend estimation via Polynomial Regression for missing or future years.
        * **Outputs:** Yield rate (tons/ha) and Total Expected Production (tons).
        """)
    with c2:
        st.subheader("Model Metrics")
        st.metric("Model Accuracy (R²)", "0.8366")
        st.metric("Mean Absolute Error (MAE)", "11.36")

# --- PREDICTION PAGE ---
elif page == "🌾 Prediction":
    st.title("🌾 Crop Yield Prediction")
    st.write("Fill in the cultivation details to generate predictions.")
    st.divider()

    left, right = st.columns(2)
    with left:
        state = st.selectbox("🌍 State", le_state.classes_)
        season = st.selectbox("🍂 Season", le_season.classes_)
        crop = st.selectbox("🌱 Crop", le_crop.classes_)

    with right:
        year = st.number_input("📅 Cultivation Year", min_value=1997, max_value=2100, value=2026, step=1)
        area = st.number_input("🌾 Area (Hectares)", min_value=0.1, value=5.0, step=0.5)

    rainfall_value = estimate_rainfall(rainfall, state, year)
    st.info(f"Estimated Rainfall for **{state}** in **{year}**: **{rainfall_value:.2f} mm**")

    st.divider()

    if st.button("🚀 Predict Crop Yield"):
        # Encoding Inputs
        state_enc = le_state.transform([state])[0]
        season_enc = le_season.transform([season])[0]
        crop_enc = le_crop.transform([crop])[0]

        # FIX: Correct column order matching exact training set: ['state', 'year', 'Season', 'Crop', 'Area', 'rainfall']
        input_df = pd.DataFrame(
            [[state_enc, year, season_enc, crop_enc, area, rainfall_value]],
            columns=["state", "year", "Season", "Crop", "Area", "rainfall"]
        )

        prediction = max(0.01, float(model.predict(input_df)[0]))
        production = prediction * area

        st.success("Prediction Completed")
        res1, res2 = st.columns(2)
        res1.metric("Predicted Yield", f"{prediction:.2f} tons/ha")
        res2.metric("Estimated Production", f"{production:.2f} tons")

        st.divider()
        if prediction < 1.0:
            st.error("⚠️ Low Yield Expected")
        elif prediction < 3.0:
            st.warning("⚡ Moderate Yield Expected")
        else:
            st.success("🌟 High Yield Expected")

# --- DASHBOARD PAGE ---
elif page == "📊 Dashboard":
    st.title("📊 Dataset & Model Dashboard")
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("States", len(le_state.classes_))
    m2.metric("Crops", len(le_crop.classes_))
    m3.metric("Seasons", len(le_season.classes_))
    m4.metric("Records", len(rainfall))

    st.divider()
    st.subheader("Historical Rainfall Analysis")
    st.line_chart(rainfall.groupby("year")["rainfall"].mean())

    st.subheader("State Rainfall Averages")
    st.bar_chart(rainfall.groupby("state")["rainfall"].mean().sort_values(ascending=False))

# --- ABOUT PAGE ---
elif page == "👤 About":
    st.title("👤 Project Details")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 👨‍🎓 Candidate")
        st.write("**Name:** Mubarak Bashir Wali")
        st.write("**Registration Number:** UG21/COMS/1132")
        st.write("**Department:** Computer Science")
        st.write("**Institution:** Aliko Dangote University of Science and Technology, Wudil")
    with col2:
        st.markdown("### 📘 Project Metadata")
        st.write("**Title:** Crop Yield Prediction System")
        st.write("**Supervisor:** Mln Mahmud Muhammad")
        st.write("**Model:** XGBoost Regressor")
        st.write("**Session:** 2026")
