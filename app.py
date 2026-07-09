import streamlit as st
import pandas as pd
import numpy as np
import pickle

from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp{
    background-color:#f5f9f6;
}

.main-title{
    font-size:42px;
    font-weight:bold;
    color:#145A32;
    text-align:center;
}

.sub-title{
    text-align:center;
    color:#555555;
    font-size:18px;
}

.card{

    background:white;

    padding:25px;

    border-radius:15px;

    box-shadow:0px 5px 15px rgba(0,0,0,0.15);

}

.success-card{

    background:#D5F5E3;

    padding:20px;

    border-radius:12px;

}

.about-card{

    background:white;

    padding:20px;

    border-radius:15px;

    box-shadow:0px 5px 15px rgba(0,0,0,0.12);

}

.stButton>button{

    width:100%;

    height:50px;

    border-radius:10px;

    font-size:18px;

    font-weight:bold;

    background:#1E8449;

    color:white;

}

.stButton>button:hover{

    background:#145A32;

    color:white;

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION
# =====================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =====================================================
# LOGIN DETAILS
# =====================================================

USERNAME = "admin"
PASSWORD = "crop123"

# =====================================================
# LOGIN PAGE
# =====================================================

def login():

    st.markdown(
        "<h1 class='main-title'>🌾 Crop Yield Prediction System</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='sub-title'>Machine Learning Based Agricultural Decision Support System</p>",
        unsafe_allow_html=True
    )

    st.write("")

    left, center, right = st.columns([1,2,1])

    with center:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("🔐 Login")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if username == USERNAME and password == PASSWORD:

                st.session_state.logged_in = True

                st.rerun()

            else:

                st.error("Invalid Username or Password")

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_files():

    try:

        model = XGBRegressor()

        model.load_model("xgb_crop_yield_model.json")

        with open("le_state.pkl","rb") as f:
            le_state = pickle.load(f)

        with open("le_crop.pkl","rb") as f:
            le_crop = pickle.load(f)

        with open("le_season.pkl","rb") as f:
            le_season = pickle.load(f)

        rainfall = pd.read_csv("rainfall_cleaned.csv")

        rainfall.columns = rainfall.columns.str.lower()

        return (
            model,
            le_state,
            le_crop,
            le_season,
            rainfall
        )

    except Exception as e:

        st.error(f"Unable to load project files.\n\n{e}")

        st.stop()

# =====================================================
# LOGIN CHECK
# =====================================================

if not st.session_state.logged_in:

    login()

    st.stop()

# =====================================================
# LOAD PROJECT FILES
# =====================================================

model, le_state, le_crop, le_season, rainfall = load_files()

# =====================================================
# RAINFALL ESTIMATION
# =====================================================

def estimate_rainfall(state, year):

    data = rainfall[
        rainfall["state"] == state
    ].copy()

    data = data.dropna(
        subset=["rainfall"]
    )

    if data.empty:

        return float(
            rainfall["rainfall"].mean()
        )

    if year in data["year"].values:

        return float(

            data.loc[
                data["year"] == year,
                "rainfall"
            ].iloc[0]

        )

    X = data[["year"]]

    y = data["rainfall"]

    poly = PolynomialFeatures(
        degree=2
    )

    X_poly = poly.fit_transform(X)

    reg = LinearRegression()

    reg.fit(
        X_poly,
        y
    )

    prediction = reg.predict(

        poly.transform([[year]])

    )[0]

    prediction = max(
        prediction,
        0
    )

    return float(prediction)
# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/wheat.png",
    width=80
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "",

    [

        "🏠 Home",

        "🌾 Prediction",

        "📊 Dashboard",

        "👤 About"

    ]

)

st.sidebar.write("")

if st.sidebar.button("Logout"):

    st.session_state.logged_in = False

    st.rerun()

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.markdown(
        "<h1 class='main-title'>🌾 Crop Yield Prediction System</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='sub-title'>Machine Learning Based Agricultural Decision Support System</p>",
        unsafe_allow_html=True
    )

    st.write("")

    a,b = st.columns(2)

    with a:

        st.metric(
            "Machine Learning Model",
            "XGBoost"
        )

    with b:

        st.metric(
            "Prediction",
            "Crop Yield"
        )

    st.divider()

    c1,c2,c3 = st.columns(3)

    with c1:

        st.success("🌾 Crop Prediction")

    with c2:

        st.info("🌧 Rainfall Estimation")

    with c3:

        st.warning("📈 Smart Agriculture")

    st.divider()

    left,right = st.columns([2,1])

    with left:

        st.subheader("Project Overview")

        st.write("""

This application predicts crop yield using an XGBoost Machine Learning model.

The prediction considers:

• State

• Crop

• Season

• Cultivation Year

• Area

• Estimated Rainfall

Future years are also supported.

Whenever rainfall data is unavailable, the application estimates rainfall using historical rainfall trends.

The output includes both:

• Predicted Yield (tons/hectare)

• Estimated Production (tons)

""")

    with right:

        st.markdown(
            "<div class='success-card'>",
            unsafe_allow_html=True
        )

        st.subheader("Navigation")

        st.write("🏠 Home")

        st.write("🌾 Prediction")

        st.write("📊 Dashboard")

        st.write("👤 About")

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        st.metric(
            "Model Accuracy (R²)",
            "0.8795"
        )

    with col2:

        st.metric(
            "Mean Absolute Error",
            "11.92"
        )

    st.success(
        "Select Prediction from the sidebar to begin."
    )

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🌾 Prediction":

    st.title("🌾 Crop Yield Prediction")

    st.write(
        "Provide the cultivation information below."
    )

    st.divider()

    left,right = st.columns(2)

    with left:

        state = st.selectbox(

            "🌍 State",

            le_state.classes_

        )

        crop = st.selectbox(

            "🌱 Crop",

            le_crop.classes_

        )

        season = st.selectbox(

            "🍂 Season",

            le_season.classes_

        )

    with right:

        year = st.number_input(

            "📅 Cultivation Year",

            min_value=1997,

            max_value=2100,

            value=2027,

            step=1

        )

        area = st.number_input(

            "🌾 Area (Hectares)",

            min_value=0.1,

            value=5.0,

            step=0.5

        )

    rainfall_value = estimate_rainfall(

        state,

        year

    )

    st.info(

        f"Estimated Rainfall : {rainfall_value:.2f} mm"

    )

    st.divider()

    if st.button("🚀 Predict Crop Yield"):

        state_enc = le_state.transform([state])[0]

        crop_enc = le_crop.transform([crop])[0]

        season_enc = le_season.transform([season])[0]

        input_data = np.array([[
            state_enc,
            year,
            season_enc,
            crop_enc,
            area,
            rainfall_value
        ]])

        prediction = max(0.01, float(model.predict(input_data)[0]))

        production = prediction * area

        st.success(
            "Prediction Completed Successfully"
        )

        c1,c2 = st.columns(2)

        with c1:

            st.metric(

                "Predicted Yield",

                f"{prediction:.2f} tons/hectare"

            )

        with c2:

            st.metric(

                "Estimated Production",

                f"{production:.2f} tons"

            )

        st.divider()

        st.subheader("Yield Status")

        if prediction < 1:

            st.error(
                "Low Yield Expected"
            )

        elif prediction < 3:

            st.warning(
                "Moderate Yield Expected"
            )

        else:

            st.success(
                "High Yield Expected"
            )

        with st.expander("Prediction Summary"):

            st.write(f"State : {state}")

            st.write(f"Crop : {crop}")

            st.write(f"Season : {season}")

            st.write(f"Cultivation Year : {year}")

            st.write(f"Area : {area:.2f} hectares")

            st.write(f"Estimated Rainfall : {rainfall_value:.2f} mm")

            st.write(f"Predicted Yield : {prediction:.2f} tons/hectare")

            st.write(f"Estimated Production : {production:.2f} tons")

# =====================================================
# DASHBOARD PAGE
# =====================================================

elif page == "📊 Dashboard":

    st.title("📊 Dashboard")

    st.write(
        "Project statistics and dataset overview."
    )

    st.divider()

    # ==========================================
    # Dataset Statistics
    # ==========================================

    total_states = len(le_state.classes_)

    total_crops = len(le_crop.classes_)

    total_seasons = len(le_season.classes_)

    total_records = len(rainfall)

    min_year = int(rainfall["year"].min())

    max_year = int(rainfall["year"].max())

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "States",
            total_states
        )

    with c2:
        st.metric(
            "Crops",
            total_crops
        )

    with c3:
        st.metric(
            "Seasons",
            total_seasons
        )

    with c4:
        st.metric(
            "Rainfall Records",
            total_records
        )

    st.divider()

    # ==========================================
    # Project Information
    # ==========================================

    left, right = st.columns([2,1])

    with left:

        st.subheader("Machine Learning Model")

        st.info("Model Used : XGBoost Regressor")

        st.success("R² Score : 0.8795")

        st.success("Mean Absolute Error : 11.92")

        st.write("Prediction Target : Crop Yield")

        st.write("Output Unit : Tons per Hectare")

    with right:

        st.subheader("Dataset")

        st.write(f"Years Available : {min_year} - {max_year}")

        st.write(f"Total States : {total_states}")

        st.write(f"Total Crops : {total_crops}")

        st.write(f"Total Seasons : {total_seasons}")

    st.divider()

    # ==========================================
    # Rainfall Statistics
    # ==========================================

    st.subheader("Rainfall Statistics")

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Average Rainfall",
            f"{rainfall['rainfall'].mean():.2f} mm"
        )

    with r2:

        st.metric(
            "Maximum Rainfall",
            f"{rainfall['rainfall'].max():.2f} mm"
        )

    with r3:

        st.metric(
            "Minimum Rainfall",
            f"{rainfall['rainfall'].min():.2f} mm"
        )

    st.divider()

    # ==========================================
    # Rainfall Trend
    # ==========================================

    st.subheader("Historical Rainfall Trend")

    yearly = rainfall.groupby("year")["rainfall"].mean()

    st.line_chart(yearly)

    st.divider()

    # ==========================================
    # State Distribution
    # ==========================================

    st.subheader("Rainfall by State")

    state_mean = (
        rainfall.groupby("state")["rainfall"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(state_mean)

    st.divider()

    # ==========================================
    # Dataset Preview
    # ==========================================

    st.subheader("Dataset Preview")

    st.dataframe(
        rainfall.head(20),
        use_container_width=True
    )
# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "👤 About":

    st.title("👤 About This Project")

    st.divider()

    st.markdown(
        """
### 🌾 Crop Yield Prediction Using Machine Learning

This application was developed as an undergraduate final year project in the
Department of Computer Science.

The system uses an XGBoost Machine Learning model to predict crop yield based on
historical agricultural and rainfall data.
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

        st.markdown("### 📘 Project Information")

        st.success("Project Title")

        st.write("Crop Yield Prediction Using Machine Learning")

        st.success("Supervisor")

        st.write("Mln Mahmud Muhammad")

        st.success("Model")

        st.write("XGBoost Regressor")

        st.success("Programming Language")

        st.write("Python")

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown("""
- Predict crop yield accurately using Machine Learning.

- Assist farmers and agricultural stakeholders in decision making.

- Estimate rainfall for future cultivation years using historical rainfall trends.

- Improve agricultural productivity through data-driven prediction.

- Provide an easy-to-use graphical interface for prediction.
""")

    st.divider()

    st.subheader("📌 Scope of the Project")

    st.write("""

The system predicts crop yield using:

- State

- Crop

- Season

- Cultivation Year

- Area

- Estimated Rainfall

The application also supports future cultivation years by estimating rainfall
using historical rainfall patterns through Polynomial Regression.

""")

    st.divider()

    st.subheader("⭐ Main Features")

    feature1, feature2 = st.columns(2)

    with feature1:

        st.success("✔ Secure Login")

        st.success("✔ Future Year Prediction")

        st.success("✔ Rainfall Estimation")

        st.success("✔ Crop Yield Prediction")

    with feature2:

        st.success("✔ Dashboard")

        st.success("✔ Dataset Statistics")

        st.success("✔ Modern User Interface")

        st.success("✔ Prediction Summary")

    st.divider()

    st.subheader("📖 User Guide")

    st.markdown("""

1. Login using the correct username and password.

2. Select **Prediction** from the sidebar.

3. Enter the cultivation information.

4. Click **Predict Crop Yield**.

5. Review the prediction results and estimated production.

6. Explore the Dashboard for statistics and charts.

""")

    st.divider()

    st.markdown(
        """
<div style='text-align:center;
padding:20px;
background:#D5F5E3;
border-radius:12px;'>

<h4>Crop Yield Prediction System</h4>

<p>Developed by <b>Mubarak Bashir Wali</b></p>

<p>Department of Computer Science</p>

<p>Faculty of Computing and Mathematical Science</p>

<p>Aliko Dangote University of Science and Technology, Wudil</p>

<p><b>2026</b></p>

</div>
""",
        unsafe_allow_html=True
    )
