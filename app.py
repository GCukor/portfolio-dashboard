import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(
    page_title="Gunnar's Data Science Portfolio",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("📊 Portfolio Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select a Project",
    ["🏠 Home", "🌍 World Happiness", "🏨 Hotel Cancellation", "🚢 Hampton Roads AIS"]
)

# Home page
if page == "🏠 Home":
    st.title("Gunnar Cukor — Data Science Portfolio")
    st.markdown("---")
    st.write("""
    Welcome to my interactive data science portfolio.
    Use the sidebar to explore three end-to-end data science projects
    covering exploratory analysis, machine learning, and time series forecasting.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🌍 World Happiness")
        st.write(
            "Exploratory analysis of global happiness across 150+ countries from 2015-2019.")
    with col2:
        st.subheader("🏨 Hotel Cancellation")
        st.write(
            "Machine learning classifier predicting hotel booking cancellations with 87% accuracy.")
    with col3:
        st.subheader("🚢 Hampton Roads AIS")
        st.write(
            "Time series forecasting of vessel traffic in one of the world's busiest naval ports.")

elif page == "🌍 World Happiness":
    st.title("🌍 World Happiness Report")
    st.markdown(
        "Exploratory analysis of global happiness across 150+ countries from 2015–2019.")
    st.markdown("---")

    @st.cache_data
    def load_happiness():
        years = [2015, 2016, 2017, 2018, 2019]
        dfs = []
        for year in years:
            df = pd.read_csv(f"data/{year}.csv")
            df["year"] = year
            df.columns = df.columns.str.strip()
            rename_map = {
                "Country or region": "country", "Country": "country",
                "Happiness Score": "happiness_score", "Happiness.Score": "happiness_score", "Score": "happiness_score",
                "Happiness Rank": "rank", "Happiness.Rank": "rank", "Overall rank": "rank",
                "Economy (GDP per Capita)": "gdp", "Economy..GDP.per.Capita.": "gdp", "GDP per capita": "gdp",
                "Family": "social_support", "Social support": "social_support",
                "Health (Life Expectancy)": "life_expectancy", "Health..Life.Expectancy.": "life_expectancy", "Healthy life expectancy": "life_expectancy",
                "Freedom": "freedom", "Freedom to make life choices": "freedom",
                "Trust (Government Corruption)": "corruption", "Trust..Government.Corruption.": "corruption", "Perceptions of corruption": "corruption",
                "Generosity": "generosity",
                "Region": "region",
            }
            df = df.rename(columns=rename_map)
            df = df.loc[:, ~df.columns.duplicated()]
            keep = ["country", "year", "rank", "happiness_score", "gdp",
                    "social_support", "life_expectancy", "freedom",
                    "corruption", "generosity", "region"]
            df = df[[col for col in keep if col in df.columns]]
            dfs.append(df)
        happiness = pd.concat(dfs, ignore_index=True)
        happiness["corruption"] = happiness["corruption"].fillna(
            happiness["corruption"].median())
        region_map = happiness[happiness["region"].notna()].groupby("country")[
            "region"].first().to_dict()
        happiness["region"] = happiness["region"].fillna(
            happiness["country"].map(region_map))
        happiness = happiness.dropna(subset=["region"])
        return happiness

    happiness = load_happiness()

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox(
            "Select Year", [2015, 2016, 2017, 2018, 2019])
    with col2:
        selected_region = st.multiselect("Filter by Region", happiness["region"].unique(
        ), default=happiness["region"].unique())

    filtered = happiness[(happiness["year"] == selected_year) & (
        happiness["region"].isin(selected_region))]

    st.subheader(f"Top 10 Happiest Countries ({selected_year})")
    top10 = filtered.nlargest(10, "happiness_score")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top10["country"], top10["happiness_score"], color="steelblue")
    ax.set_xlabel("Happiness Score")
    st.pyplot(fig)

    st.subheader("What Factors Drive Happiness?")
    corr_cols = ["happiness_score", "gdp", "social_support",
                 "life_expectancy", "freedom", "corruption", "generosity"]
    available_cols = [col for col in corr_cols if col in filtered.columns]
    corr_matrix = filtered[available_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, square=True, ax=ax2)
    st.pyplot(fig2)

elif page == "🏨 Hotel Cancellation":
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.markdown(
        "Machine learning classifier predicting hotel booking cancellations with 87% accuracy.")
    st.markdown("---")

    @st.cache_resource
    def load_model():
        df = pd.read_csv("data/hotel_bookings.csv")
        df["children"] = df["children"].fillna(0)
        df["agent"] = df["agent"].fillna(0)
        df.drop(columns=["company", "reservation_status",
                "reservation_status_date", "country"], inplace=True)
        cols_to_encode = ["meal", "market_segment", "distribution_channel",
                          "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type"]
        df["hotel"] = df["hotel"].map({"City Hotel": 0, "Resort Hotel": 1})
        month_map = {"January": 1, "February": 2, "March": 3, "April": 4,
                     "May": 5, "June": 6, "July": 7, "August": 8,
                     "September": 9, "October": 10, "November": 11, "December": 12}
        df["arrival_date_month"] = df["arrival_date_month"].map(month_map)
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
        X = df.drop(columns=["is_canceled"])
        y = df["is_canceled"]
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return model, X.columns.tolist()

    model, feature_cols = load_model()

    st.subheader("🔮 Will This Booking Be Cancelled?")
    st.write("Adjust the booking details below to get a cancellation prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        lead_time = st.slider("Lead Time (days)", 0, 500, 100)
        adr = st.slider("Average Daily Rate ($)", 0, 500, 100)
        total_special_requests = st.slider("Special Requests", 0, 5, 0)
    with col2:
        stays_in_weekend_nights = st.slider("Weekend Nights", 0, 10, 1)
        stays_in_week_nights = st.slider("Week Nights", 0, 10, 2)
        previous_cancellations = st.slider("Previous Cancellations", 0, 10, 0)
    with col3:
        deposit_type = st.selectbox(
            "Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
        customer_type = st.selectbox(
            "Customer Type", ["Transient", "Contract", "Group", "Transient-Party"])
        hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])

    input_data = pd.DataFrame(columns=feature_cols)
    input_data.loc[0] = 0
    input_data["lead_time"] = lead_time
    input_data["adr"] = adr
    input_data["total_of_special_requests"] = total_special_requests
    input_data["stays_in_weekend_nights"] = stays_in_weekend_nights
    input_data["stays_in_week_nights"] = stays_in_week_nights
    input_data["previous_cancellations"] = previous_cancellations
    input_data["hotel"] = 0 if hotel == "City Hotel" else 1

    deposit_col = f"deposit_type_{deposit_type.replace(' ', '_')}"
    if deposit_col in feature_cols:
        input_data[deposit_col] = 1

    customer_col = f"customer_type_{customer_type.replace('-', '_')}"
    if customer_col in feature_cols:
        input_data[customer_col] = 1

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(
            f"⚠️ High Cancellation Risk — {probability:.0%} probability of cancellation")
    else:
        st.success(
            f"✅ Low Cancellation Risk — {probability:.0%} probability of cancellation")

elif page == "🚢 Hampton Roads AIS":
    st.title("🚢 Hampton Roads Vessel Traffic")
    st.markdown(
        "Time series analysis of vessel traffic near Naval Station Norfolk (2023).")
    st.markdown("---")

    @st.cache_data
    def load_ais():
        df = pd.read_csv("data/hampton_roads_ais.csv", low_memory=False)
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
        df["hour"] = df["BaseDateTime"].dt.hour
        df["day_of_week"] = df["BaseDateTime"].dt.dayofweek
        df["month"] = df["BaseDateTime"].dt.month
        return df

    df = load_ais()

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total AIS Pings", f"{len(df):,}")
    with col2:
        st.metric("Unique Vessels", f"{df['MMSI'].nunique():,}")
    with col3:
        st.metric("Date Range", "Jan 2023 – Jan 2024")

    st.markdown("---")

    # Traffic by hour
    st.subheader("Vessel Traffic by Hour of Day")
    hourly = df.groupby("hour")["MMSI"].nunique()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(hourly.index, hourly.values, color="steelblue")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Unique Vessels")
    st.pyplot(fig)

    # Traffic by month
    st.subheader("Vessel Traffic by Month")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = df.groupby("month")["MMSI"].nunique()
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(month_labels, monthly.values, color="steelblue")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Unique Vessels")
    st.pyplot(fig2)
