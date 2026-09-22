import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

# =====================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
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
    .main-title { font-size: 40px; font-weight: bold; color: #145A32; text-align: center; }
    .sub-title { text-align: center; color: #555555; font-size: 17px; margin-bottom: 20px; }
    .card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0px 5px 15px rgba(0,0,0,0.12); }
    .success-card { background: #D5F5E3; padding: 20px; border-radius: 12px; }
    .stButton>button { width: 100%; height: 50px; border-radius: 10px; font-size: 18px; font-weight: bold; background: #1E8449; color: white; }
    .stButton>button:hover { background: #145A32; color: white; }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize Session State for Login
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
        st.error(f"Unable to load project files.\n\n{e}")
        st.stop()


def estimate_rainfall(rainfall_df, state, year):
    data = rainfall_df[rainfall_df["state"] == state].dropna(subset=["rainfall"])

    if data.empty:
        return float(rainfall_df["rainfall"].mean())

    if year in data["year"].values:
        return float(data.loc[data["year"] == year, "rainfall"].iloc[0])

    # Polynomial extrapolation for future/missing years
    X = data[["year"]]
    y = data["rainfall"]
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    reg = LinearRegression()
    reg.fit(X_poly, y)

    pred = reg.predict(poly.transform([[year]]))[0]
    return float(max(pred, 0))


# =====================================================
# 3. LOGIN PAGE
# =====================================================
def login_screen():
    st.markdown("<h1 class='main-title'>🌾 Crop Yield Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Machine Learning Based Agricultural Decision Support System</p>", unsafe_allow_html=True)

    st.write("")
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

# Load model and datasets after authentication
model, le_state, le_crop, le_season, rainfall = load_assets()

# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================
st.sidebar.image("https://img.icons8.com/color/96/wheat.png", width=80)
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

    st.write("")
    a, b = st.columns(2)
    a.metric("Machine Learning Model", "XGBoost Regressor")
    b.metric("Prediction Target", "Crop Yield & Production")

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.success("🌾 Crop Prediction")
    c2.info("🌧 Rainfall Estimation")
    c3.warning("📈 Smart Agriculture")

    st.divider()

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Project Overview")
        st.write("""
        This application predicts crop yield using an **XGBoost Machine Learning** model based on:
        
        • **State**
        • **Crop Type**
        • **Season**
        • **Cultivation Year**
        • **Area (Hectares)**
        • **Estimated Rainfall (mm)**

        Whenever historical rainfall data for a future year is unavailable, the application automatically estimates it using historical rainfall trends.
        """)
    with right:
        st.markdown("<div class='success-card'>", unsafe_allow_html=True)
        st.subheader("Navigation")
        st.write("🏠 Home")
        st.write("🌾 Prediction")
        st.write("📊 Dashboard")
        st.write("👤 About")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy (R²)", "0.8795")
    col2.metric("Mean Absolute Error (MAE)", "11.92")

    st.success("Select **Prediction** from the sidebar to begin.")

# --- PREDICTION PAGE ---
elif page == "🌾 Prediction":
    st.title("🌾 Crop Yield Prediction")
    st.write("Provide the cultivation information below to calculate predictions.")
    st.divider()

    left, right = st.columns(2)
    with left:
        state = st.selectbox("🌍 State", le_state.classes_)
        crop = st.selectbox("🌱 Crop", le_crop.classes_)
        season = st.selectbox("🍂 Season", le_season.classes_)

    with right:
        year = st.number_input("📅 Cultivation Year", min_value=1997, max_value=2100, value=2027, step=1)
        area = st.number_input("🌾 Area (Hectares)", min_value=0.1, value=5.0, step=0.5)

    # Estimate Rainfall based on State and Year
    rainfall_value = estimate_rainfall(rainfall, state, year)
    st.info(f"Estimated Rainfall : **{rainfall_value:.2f} mm**")

    st.divider()

    if st.button("🚀 Predict Crop Yield"):
        # Encode inputs using saved LabelEncoders
        state_enc = le_state.transform([state])[0]
        season_enc = le_season.transform([season])[0]
        crop_enc = le_crop.transform([crop])[0]

        # Construct DataFrame with EXACT column names and ordering matching model training
        input_df = pd.DataFrame(
            [[state_enc, year, season_enc, crop_enc, area, rainfall_value]],
            columns=["state", "year", "Season", "Crop", "Area", "rainfall"]
        )

        # Generate accurate XGBoost prediction
        prediction = max(0.01, float(model.predict(input_df)[0]))
        production = prediction * area

        st.success("Prediction Completed Successfully")

        c1, c2 = st.columns(2)
        c1.metric("Predicted Yield", f"{prediction:.2f} tons/hectare")
        c2.metric("Estimated Production", f"{production:.2f} tons")

        st.divider()

        st.subheader("Yield Status")
        if prediction < 1.0:
            st.error("⚠️ Low Yield Expected")
        elif prediction < 3.0:
            st.warning("⚡ Moderate Yield Expected")
        else:
            st.success("🌟 High Yield Expected")

        with st.expander("🔍 Prediction Summary Details"):
            st.write(f"**State:** {state}")
            st.write(f"**Crop:** {crop}")
            st.write(f"**Season:** {season}")
            st.write(f"**Cultivation Year:** {year}")
            st.write(f"**Area:** {area:.2f} hectares")
            st.write(f"**Estimated Rainfall:** {rainfall_value:.2f} mm")
            st.write(f"**Predicted Yield:** {prediction:.2f} tons/hectare")
            st.write(f"**Estimated Production:** {production:.2f} tons")

# --- DASHBOARD PAGE ---
elif page == "📊 Dashboard":
    st.title("📊 Project Dashboard")
    st.write("Dataset statistics and model insights.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States", len(le_state.classes_))
    c2.metric("Crops", len(le_crop.classes_))
    c3.metric("Seasons", len(le_season.classes_))
    c4.metric("Rainfall Records", len(rainfall))

    st.divider()

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Machine Learning Model Statistics")
        st.info("Model Used : XGBoost Regressor")
        st.success("R² Score : 0.8795")
        st.success("Mean Absolute Error : 11.92")
    with right:
        st.subheader("Dataset Scope")
        st.write(f"Years Available: {int(rainfall['year'].min())} - {int(rainfall['year'].max())}")
        st.write(f"Total States: {len(le_state.classes_)}")
        st.write(f"Total Crops: {len(le_crop.classes_)}")

    st.divider()

    st.subheader("Historical Rainfall Trend")
    st.line_chart(rainfall.groupby("year")["rainfall"].mean())

    st.subheader("Average Rainfall by State")
    st.bar_chart(rainfall.groupby("state")["rainfall"].mean().sort_values(ascending=False))

    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(rainfall.head(20), use_container_width=True)

# --- ABOUT PAGE ---
elif page == "👤 About":
    st.title("👤 About This Project")
    st.divider()

    st.markdown(
        """
### 🌾 Crop Yield Prediction Using Machine Learning
This application was developed as an undergraduate final year project in the **Department of Computer Science**.
The system leverages an **XGBoost Regressor** model to predict crop yield based on historical agricultural parameters and weather trends.
"""
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 👨‍🎓 Student Information")
        st.info("**Name:** Mubarak Bashir Wali")
        st.info("**Matric Number:** UG21/COMS/1132")
        st.info("**Department:** Computer Science")
        st.info("**Faculty:** Faculty of Computing and Mathematical Science")
        st.info("**University:** Aliko Dangote University of Science and Technology, Wudil")
        st.info("**Year:** 2026")

    with col2:
        st.markdown("### 📘 Project Metadata")
        st.success("**Project Title:** Crop Yield Prediction System")
        st.write("**Supervisor:** Mln Mahmud Muhammad")
        st.write("**Model:** XGBoost Regressor")
        st.write("**Programming Language:** Python")

    st.divider()

    st.subheader("⭐ Key Features")
    f1, f2 = st.columns(2)
    with f1:
        st.success("✔ Secure Authentication")
        st.success("✔ Polynomial Rainfall Forecasting")
    with f2:
        st.success("✔ XGBoost Crop Yield Prediction")
        st.success("✔ Interactive Analytical Dashboard")
