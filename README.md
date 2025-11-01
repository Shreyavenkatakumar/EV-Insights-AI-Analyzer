# 🚗 Electric Vehicle (EV) Sales Prediction

### 📘 Overview
This project is developed as part of the **AICTE – EduNet Internship (Electric Vehicle Theme)**.  
It focuses on **analyzing and predicting global Electric Vehicle (EV) sales trends** using machine learning and exploratory data analysis techniques.

---

### 📊 Project Highlights
- **Dataset:** EV sales data from multiple countries (2010–2023)  
- **Tech Stack:** Python, Pandas, Matplotlib, Seaborn  
- **Notebook:** `week1_ev_sales_prediction.ipynb`  
- **Visualizations:**
  - Global EV Sales Trend  
  - Top 5 Regions by EV Sales  
  - Correlation Heatmap  

---

### 🧹 Data Cleaning
Steps performed before analysis:
1. Filtered only EV sales-related data (`parameter == "EV sales"`)
2. Selected relevant columns: `region`, `year`, and `value`
3. Converted columns to numeric values and removed missing data
4. Verified the cleaned dataset shape → `(1342, 3)`

---

### 📈 Exploratory Data Analysis (EDA)
#### 1️⃣ Global EV Sales Over the Years
![EV Sales Trend](images/ev_sales_trend.png)

#### 2️⃣ Top 5 Regions by Total EV Sales
![Top Regions](images/top_regions.png)

#### 3️⃣ Correlation Heatmap
![Heatmap](images/heatmap.png)

---

### 🧠 Insights
- EV sales have shown **steady growth after 2015**, accelerating rapidly post-2020.  
- **China, Europe, and the USA** contribute the most to global EV adoption.  
- Strong correlation between year and sales value confirms consistent market expansion.

---

### 💻 Tools & Libraries Used
- **Python 3.x**
- **Pandas** – for data preprocessing  
- **Matplotlib & Seaborn** – for data visualization  
- **Jupyter Notebook** – for analysis and documentation  

---

### 🏁 Future Scope
- Implement predictive modeling using **Linear Regression / Random Forest**  
- Expand analysis to include **EV stock share, powertrain, and emission data**  
- Deploy a simple dashboard for **interactive EV trend visualization**

---

### 👩‍💻 Author
**Shreya V**  
B.E. Computer Science and Engineering (Cybersecurity)  
Sri Krishna College of Technology, Coimbatore  
🔗 [GitHub Profile](https://github.com/Shreyavenkatakumar)

---

⭐ *If you like this project, don’t forget to star the repository!*
