# AMCAT Employment Outcome - Data Analysis

A complete Exploratory Data Analysis (EDA) of the Aspiring Minds Employment Outcome 2015 (AMEO) dataset, exploring the relationship between engineering graduates' education, skills, and employment outcomes.

## Dataset
- **Source:** [Kaggle - Aspiring Minds Employment Outcome 2015](https://www.kaggle.com/datasets/aspiringmindsdataset/aspiringmindsemploymentoutcome2015)
- **Size:** ~4,000 engineering graduate records
- **Features:** 39 columns including salary, job role, college tier, AMCAT test scores, personality traits

## What This Project Covers

### 1. Data Cleaning
- Date column conversion and formatting
- Outlier detection using IQR method
- Outlier treatment for salary, domain scores, and personality traits

### 2. Feature Engineering
- Job role consolidation from 400+ raw designations into 9 clean categories
- Specialization mapping from 46 branches into 6 groups (CSE, ECE, EEE, MECH, CE, Other)

### 3. Univariate Analysis
- Statistical summary of all numerical and categorical columns
- Frequency distributions and histograms for all features
- Boxplots for outlier visualization

### 4. Bivariate Analysis
- Salary vs Job Role
- Salary vs College Tier
- Salary vs Specialization
- Salary vs Degree
- Salary vs Gender
- Correlation heatmap and pairplot

### 5. Research Questions Answered
- **Q1:** Do CSE fresh graduates earn 2.5–3 LPA as commonly claimed?
- **Q2:** Is there a relationship between gender and specialization choice?

## Key Findings

| Finding | Result |
|---|---|
| Highest earning role | System Engineer (Mean: 3.62L) |
| College tier impact | Tier-1 earns ~50% more than Tier-2 |
| Best specialization for salary | CSE (Mean: 3.12L) |
| Fresh graduate salary claim (2.5-3L) | **Not supported** by data |
| Gender vs specialization | **No significant relationship** found |

## Tech Stack
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/172005-dot/Amcat-project

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# 3. Download the dataset from Kaggle and place AMCAT.csv in the project folder

# 4. Run the notebook
jupyter notebook amcat_analysis.ipynb
```

## Project Structure
```
Amcat-project/
├── amcat_analysis.py       # Main analysis code
├── AMCAT.csv               # Dataset (download from Kaggle)
└── README.md
```

## Author
**Badveli Chandrika** | B.Tech CS & Data Science | G Pulla Reddy Engineering College
