# DAMajorproject--Movie-Analytics.github.io
# 🎬 Bollywood Box Office Success Factor Analysis

A data-driven project analyzing the key factors that influence the commercial success of Bollywood movies. This includes exploratory data analysis (EDA), visual dashboards, and predictive modeling using Python and Power BI.

---

## 📌 Project Overview

This project aims to:
- Identify the primary factors influencing box office collections.
- Analyze the impact of genre, budget, ratings, runtime, release timing, actors, and directors.
- Build a predictive model to estimate box office revenue before a film's release.
- Provide interactive dashboards for business stakeholders.

---

## 📁 Contents

├── bollywood_movie_analytics.csv # Movie dataset
├── ProjDA.py # Python script for analysis and modeling
├── PowerBI Dashboard (Screenshot).png # Dashboard preview
├── Movie_Analytics_Presentation.pptx # Final presentation
├── README.md # Project documentation


---

## 🔍 Exploratory Data Analysis (EDA)

The analysis covers:
- Distribution of box office collections
- Relationship between budget and revenue
- IMDb & TMDB ratings vs. box office
- Genre and language performance
- Seasonal trends by release month
- Runtime and collection patterns
- Correlation heatmap

Visualizations created using:

- `matplotlib`, `seaborn`, `pandas`

---

## 📊 Dashboard (Power BI)

The Power BI dashboard includes:
- Genre-wise Mojo Score
- Director-wise performance by month
- Budget vs. Box Office by genre
- Language and genre heatmaps
- Ratings and audience feedback by genre

👉 View screenshot in `Screenshot 2025-05-30 111043.png`

---

## 🤖 Predictive Modeling

- **Model Used:** Linear Regression
- **Features:** Budget, IMDb Rating, TMDB Rating, Runtime, Genre (encoded)
- **Target:** BoxOffice_INR
- **Evaluation Metrics:**
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)

> 📈 The model performs moderately well in predicting average-range movies but underestimates blockbusters, suggesting potential for future improvements using more advanced models.

---

## 📌 Key Insights

- 🎬 Thriller & Drama genres earn the highest revenue.
- 💰 Higher budgets generally lead to better earnings.
- 🌟 Top directors and actors significantly boost revenue.
- 📆 Certain months like May, April, and November show slightly stronger box office performance.
- 🎯 Ratings (IMDb, TMDB) positively correlate with success.
- 📉 Runtime shows weak correlation with revenue.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Libraries:** pandas, numpy, seaborn, matplotlib, scikit-learn
- **Visualization:** Power BI
- **Model:** Linear Regression

---

🤝 Acknowledgements
IMDb & TMDB for ratings reference

Publicly available Bollywood datasets

Inspiration from real-world entertainment analytics.

Thank You!
