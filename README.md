⚡ EV Insights AI Analyzer
AI-powered Dashboard for EV Sales Analysis, Prediction, and Interactive Insights
<div align="center">

🚗 Smart Analytics • 🤖 Offline AI Chatbot • 📊 Interactive Visuals • 🔮 ML Predictions
A complete EV analytics powerhouse built using Streamlit.

</div>
🔥 Overview

EV Insights AI Analyzer is a powerful, interactive web application designed to analyze global Electric Vehicle (EV) sales data.
It combines data analytics, machine learning prediction, offline AI chat, and visual storytelling in one beautifully organized dashboard.

Built with Python + Streamlit, the tool helps users:

✔ Explore historical EV sales
✔ Predict future trends
✔ Interact with an intelligent data chatbot
✔ Upload and analyze their own datasets
✔ Visualize top regions, trends, and patterns

Perfect for projects, hackathons, academic submissions, research, and live demos.

✨ Key Features
⚡ 1. AI Chat Assistant (NO API Required)

A fully offline chatbot that understands your queries and analyzes the dataset.

Ask questions like:

“What are the top 5 regions by EV sales?”

“Which year had the highest EV sales?”

“How many EVs were sold in China?”

“Tell me about EVs.”

It uses pattern recognition + rules to give intelligent, clean answers.

🔮 2. ML-Based Sales Prediction

Predict future EV sales based on:

Region

Mode

Powertrain

Category

Year

Features:

✔ Pretrained scikit-learn model
✔ Polynomial features
✔ Encoded categorical values
✔ Clean metrics & visual output

📊 3. Interactive Data Visualizations

Built using Matplotlib + Seaborn, including:

EV sales trend (year-wise)

Top 10 regions by sales

Visuals for any uploaded CSV

All visuals are clean, readable, and presented in professional container blocks.

📂 4. Upload & Analyze Your Own CSV

Upload a custom dataset with the columns:

region, year, value, parameter


The system automatically:

✔ Validates
✔ Cleans
✔ Analyzes
✔ Visualizes

your dataset.

🎨 5. Modern & Professional UI

Thanks to a custom theme in .streamlit/config.toml, the UI includes:

Soft color palette

Sidebar navigation

Section cards with borders

Typewriter animation for chatbot

Polished layout with metric cards

Looks clean, premium, and highly presentable.

🧠 Tech Stack
Component	Technology
Frontend / UI	Streamlit
Backend	Python
Machine Learning	Scikit-Learn
Visuals	Matplotlib, Seaborn
Storage	Joblib
Dataset	IEA EV Sales Dataset
📁 Project Structure
EV-Insights-AI-Analyzer/
│── app.py
│── models/
│      ├── model.pkl
│      ├── scaler.pkl
│      ├── encoders.pkl
│      ├── poly.pkl
│── images/
│      └── logo.png
│── IEA-EV-dataEV salesHistoricalCars.csv
│── .streamlit/
│      └── config.toml
│── README.md
│── requirements.txt

🚀 Running Locally
1. Clone the Repository
git clone https://github.com/your-username/EV-Insights-AI-Analyzer.git
cd EV-Insights-AI-Analyzer

2. Install Dependencies
pip install -r requirements.txt

3. Run the App
streamlit run app.py

🌐 Deployment (Streamlit Cloud)

Push this repository to GitHub

Go to https://share.streamlit.io

Connect your repo

Select app.py

Deploy 🚀

Make sure to upload the models/ folder and dataset too.

🎯 Why This Project Stands Out

Clean UI

Offline AI chatbot

Real EV dataset

Strong ML integration

Reusable components

Hackathon-ready quality

Easy to extend

This level of polish and functionality is exactly what judges love.

💡 Future Enhancements

Live EV news feed

Multi-model prediction comparison

Region-wise forecasting

Battery type / manufacturer-wise analysis

👩‍💻 Author

Shreya V
Cybersecurity Student | ML & Data Analytics Enthusiast
linkedin: https://www.linkedin.com/in/shreya-v-177672294/
Github:https://github.com/Shreyavenkatakumar
