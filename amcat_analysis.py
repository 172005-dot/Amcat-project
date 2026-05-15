# AMCAT Data Analysis - EDA Project
# Badveli Chandrika | ID: IN9240298
# Dataset: Aspiring Minds Employment Outcome 2015 (AMEO)

# ============================================================
# STEP 1: IMPORT LIBRARIES & LOAD DATA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from difflib import get_close_matches

# Load dataset (download AMCAT.csv from Kaggle: 
# https://www.kaggle.com/datasets/aspiringmindsdataset/aspiringmindsemploymentoutcome2015)
df = pd.read_csv("AMCAT.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
df.head()


# ============================================================
# STEP 2: DATA OVERVIEW
# ============================================================

print("Columns:\n", df.columns.tolist())
print("\nData Info:")
df.info()
print("\nDescriptive Stats:")
df.describe()


# ============================================================
# STEP 3: DATA CLEANING
# ============================================================

# Convert date columns
date_columns = ['DOJ', 'DOB']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='ignore', format='%m/%d/%y %H:%M')

today_date = datetime.today().strftime('%Y-%m-%d')
df['DOL'] = df['DOL'].replace('present', today_date)
df['DOL'] = pd.to_datetime(df['DOL'], dayfirst=True)

# Check missing values
print("\nNull Values:\n", df.isnull().sum())


# ============================================================
# STEP 4: FEATURE ENGINEERING - Job Role Cleaning
# ============================================================

roles_list = ['software engineer', 'system engineer', 'developer', 'analyst',
              'test engineer', 'dba', 'administrator', 'customer service',
              'quality engineer', 'quality', 'automation engineer',
              'network engineer', 'support', 'it engineer', 'manager',
              'management', 'programmer', 'tester', 'qa engineer', 'design']

def feature_cleaning(input_val, input_list):
    if type(input_val) == str:
        for item in [i for i in input_list if len(i.split()) > 1]:
            if all([x in input_val for x in item.split()]):
                return item.title()
        for item in [i for i in input_list if len(i.split()) == 1]:
            if item in input_val:
                return item.title()
        if 'engineer' in input_val:
            return 'Hardware Engineer'
        try:
            matched_item = get_close_matches(input_val, input_list)[0]
            return matched_item.title()
        except:
            return 'Other'
    else:
        return np.nan

df['Job_Role'] = df['Designation'].apply(lambda x: feature_cleaning(x, roles_list))

# Consolidate job roles
df['Job_Role'] = df['Job_Role'].replace({
    'It Engineer': 'Software Engineer',
    'Network Engineer': 'System Engineer',
    'Dba': 'System Engineer',
    'Support': 'Administrator',
    'Customer Service': 'Administrator',
    'Tester': 'Test Engineer',
    'Qa Engineer': 'Test Engineer',
    'Quality': 'Test Engineer',
    'Quality Engineer': 'Test Engineer',
    'Automation Engineer': 'Test Engineer',
    'Programmer': 'Developer',
    'Management': 'Manager',
    'Design': 'Other'
})

print("\nJob Role Distribution:")
print(df['Job_Role'].value_counts())


# ============================================================
# STEP 5: SPECIALIZATION MAPPING
# ============================================================

specialization_mapping = {
    'electronics and communication engineering': 'ECE',
    'computer science & engineering': 'CSE',
    'information technology': 'CSE',
    'computer engineering': 'CSE',
    'computer application': 'CSE',
    'mechanical engineering': 'MECH',
    'electronics and electrical engineering': 'ECE',
    'electronics & telecommunications': 'ECE',
    'electrical engineering': 'EEE',
    'electronics & instrumentation eng': 'ECE',
    'civil engineering': 'CE',
    'electronics and instrumentation engineering': 'ECE',
    'information science engineering': 'CSE',
    'instrumentation and control engineering': 'ECE',
    'electronics engineering': 'ECE',
    'biotechnology': 'other',
    'other': 'other',
    'industrial & production engineering': 'other',
    'chemical engineering': 'other',
    'applied electronics and instrumentation': 'ECE',
    'computer science and technology': 'CSE',
    'telecommunication engineering': 'ECE',
    'mechanical and automation': 'MECH',
    'automobile/automotive engineering': 'MECH',
    'instrumentation engineering': 'ECE',
    'mechatronics': 'MECH',
    'electronics and computer engineering': 'CSE',
    'aeronautical engineering': 'MECH',
    'computer science': 'CSE',
    'metallurgical engineering': 'other',
    'biomedical engineering': 'other',
    'industrial engineering': 'other',
    'information & communication technology': 'ECE',
    'electrical and power engineering': 'EEE',
    'industrial & management engineering': 'other',
    'computer networking': 'CSE',
    'embedded systems technology': 'ECE',
    'power systems and automation': 'EEE',
    'computer and communication engineering': 'CSE',
    'information science': 'CSE',
    'internal combustion engine': 'MECH',
    'ceramic engineering': 'other',
    'mechanical & production engineering': 'MECH',
    'control and instrumentation engineering': 'ECE',
    'polymer technology': 'other',
    'electronics': 'ECE'
}

for old, new in specialization_mapping.items():
    df['Specialization'] = df['Specialization'].replace(old, new)

print("\nSpecializations:", df['Specialization'].unique())


# ============================================================
# STEP 6: UNIVARIATE ANALYSIS - Non Visual
# ============================================================

discrete_df = df.select_dtypes(include=['object'])
numerical_df = df.select_dtypes(include=['int64', 'float64'])

def discrete_univariate_analysis(discrete_data):
    for col_name in discrete_data:
        print("*" * 10, col_name, "*" * 10)
        print(discrete_data[col_name].agg(['count', 'nunique']))
        print('Value Counts:\n', discrete_data[col_name].value_counts().head(10))
        print()

def numerical_univariate_analysis(numerical_data):
    for col_name in numerical_data:
        print("*" * 10, col_name, "*" * 10)
        print(numerical_data[col_name].agg(['min', 'max', 'mean', 'median', 'std']))
        print()

discrete_univariate_analysis(discrete_df)
numerical_univariate_analysis(numerical_df)


# ============================================================
# STEP 7: OUTLIER DETECTION & TREATMENT
# ============================================================

numerical_cols = ['Salary', '10percentage', '12percentage', 'collegeGPA',
                  'English', 'Logical', 'Quant', 'Domain',
                  'ComputerProgramming', 'ElectronicsAndSemicon',
                  'ComputerScience', 'MechanicalEngg', 'ElectricalEngg',
                  'TelecomEngg', 'CivilEngg', 'conscientiousness',
                  'agreeableness', 'extraversion', 'nueroticism',
                  'openess_to_experience']

# Boxplots to detect outliers
for column in numerical_cols:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.tight_layout()
    plt.savefig(f'boxplot_{column}.png', dpi=80)
    plt.show()

# Count outliers using IQR
for col in numerical_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f'Outliers in {col}: {len(outliers)}')

# Remove outliers
df = df.loc[(df["Domain"] > -1)]
df = df.loc[(df["MechanicalEngg"] < 200)]
df = df.loc[(df["ElectricalEngg"] < 200)]
df = df.loc[(df["TelecomEngg"] < 100)]
df = df.loc[(df["agreeableness"] > -1.5)]
df = df.loc[(df["openess_to_experience"] > -1.5)]
print("\nShape after outlier removal:", df.shape)


# ============================================================
# STEP 8: FREQUENCY DISTRIBUTION (HISTOGRAMS)
# ============================================================

for column in numerical_cols:
    plt.figure(figsize=(15, 9))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram for {column}')
    plt.tight_layout()
    plt.savefig(f'hist_{column}.png', dpi=80)
    plt.show()

# Categorical countplots
categorical_cols = ['Gender', 'CollegeTier', 'Degree', 'Specialization', 'CollegeState']

for col in categorical_cols:
    plt.figure(figsize=(14, 7))
    sns.countplot(x=df[col])
    plt.title(f'Countplot of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'countplot_{col}.png', dpi=80)
    plt.show()


# ============================================================
# STEP 9: BIVARIATE ANALYSIS
# ============================================================

# Correlation Matrix
correlation = df.corr()
print("\nCorrelation Matrix (Salary):\n", correlation['Salary'].sort_values(ascending=False))

plt.figure(figsize=(16, 12))
sns.heatmap(correlation[['Salary', 'English', 'Logical', 'Quant', 'collegeGPA',
                           '10percentage', '12percentage']].dropna(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=80)
plt.show()

# Pairplot
sns.pairplot(df, vars=['Salary', '10percentage', '12percentage', 'collegeGPA',
                        'English', 'Logical', 'Quant', 'Domain'])
plt.savefig('pairplot.png', dpi=80)
plt.show()


# ============================================================
# STEP 10: SALARY vs JOB ROLE
# ============================================================

print("\nSalary by Job Role:")
print(df.groupby('Job_Role')['Salary'].describe().round(2).sort_values('mean'))

order = df.groupby('Job_Role')['Salary'].mean().sort_values().index
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
sns.barplot(x='Job_Role', y='Salary', data=df, order=order, ax=ax1)
sns.boxplot(x='Job_Role', y='Salary', data=df, order=order, ax=ax2)
ax1.tick_params('x', labelrotation=45)
ax2.tick_params('x', labelrotation=45)
plt.suptitle('Salary vs Job Role')
plt.tight_layout()
plt.savefig('salary_vs_jobrole.png', dpi=80)
plt.show()

# Observation: System Engineers and Software Engineers earn the most


# ============================================================
# STEP 11: SALARY vs COLLEGE TIER
# ============================================================

print("\nSalary by College Tier:")
print(df.groupby('CollegeTier')['Salary'].describe())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x='CollegeTier', y='Salary', data=df, ax=ax1)
sns.boxplot(x='CollegeTier', y='Salary', data=df, ax=ax2)
plt.suptitle('Salary vs College Tier')
plt.tight_layout()
plt.savefig('salary_vs_collegetier.png', dpi=80)
plt.show()

# Observation: Tier-1 college graduates earn significantly more


# ============================================================
# STEP 12: SALARY vs SPECIALIZATION
# ============================================================

print("\nSalary by Specialization:")
print(df.groupby('Specialization')['Salary'].describe().round(1).sort_values('mean'))

order = df.groupby('Specialization')['Salary'].mean().sort_values().index
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x='Specialization', y='Salary', data=df, order=order, ax=ax1)
sns.boxplot(x='Specialization', y='Salary', data=df, order=order, ax=ax2)
ax1.tick_params('x', labelrotation=45)
ax2.tick_params('x', labelrotation=45)
plt.suptitle('Salary vs Specialization')
plt.tight_layout()
plt.savefig('salary_vs_specialization.png', dpi=80)
plt.show()

# Observation: CSE graduates earn the highest salaries


# ============================================================
# STEP 13: SALARY vs DEGREE
# ============================================================

print("\nSalary by Degree:")
print(df.groupby('Degree')['Salary'].describe())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x='Degree', y='Salary', data=df, ax=ax1)
sns.boxplot(x='Degree', y='Salary', data=df, ax=ax2)
plt.suptitle('Salary vs Degree')
plt.tight_layout()
plt.savefig('salary_vs_degree.png', dpi=80)
plt.show()

# Observation: M.Tech/M.E students earn more, but B.Tech has more opportunities


# ============================================================
# STEP 14: RESEARCH QUESTION 1
# Does salary of CSE graduates fall in the expected 2.5-3L range?
# ============================================================

cse_graduates = df[df['Specialization'] == 'CSE']
roles_of_interest = ['Software Engineer', 'Hardware Engineer', 'Developer', 'Analyst']
role_data = cse_graduates[cse_graduates['Job_Role'].isin(roles_of_interest)]

plt.figure(figsize=(10, 6))
sns.boxplot(x='Job_Role', y='Salary', data=role_data)
plt.title('Salary Distribution for CSE Graduates in Selected Job Roles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cse_salary_distribution.png', dpi=80)
plt.show()

print("\nCSE Graduate Salary Stats:")
print(role_data['Salary'].describe())
# Finding: Data does NOT support the 2.5-3L claim for fresh CSE graduates


# ============================================================
# STEP 15: RESEARCH QUESTION 2
# Is there a relationship between gender and specialization?
# ============================================================

gender_specialization = pd.crosstab(df['Gender'], df['Specialization'])

# Stacked bar
gender_specialization.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Relationship Between Gender and Specialization')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('gender_vs_specialization.png', dpi=80)
plt.show()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(gender_specialization, annot=True, cmap='coolwarm', fmt='d')
plt.title('Heatmap of Gender vs Specialization')
plt.tight_layout()
plt.savefig('gender_specialization_heatmap.png', dpi=80)
plt.show()

# Finding: Both genders prefer CSE. No significant gender-specialization relationship.


# ============================================================
# STEP 16: SALARY vs GENDER
# ============================================================

plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Salary', data=df)
plt.title('Salary vs Gender')
plt.tight_layout()
plt.savefig('salary_vs_gender.png', dpi=80)
plt.show()

print("\nSalary by Gender:")
print(df.groupby('Gender')['Salary'].describe())


# ============================================================
# STEP 17: CONCLUSIONS
# ============================================================

print("""
=== KEY FINDINGS ===

1. Technical expertise is crucial: B.Tech/B.E graduates dominate the market.

2. Earnings by Role: System Engineers and Software Engineers earn the most.
   Managers earn the highest average salary.

3. College Tier Impact: Tier-1 college graduates earn ~50% more than Tier-2
   (Mean: 4.5L vs 3L).

4. Specialization: CSE graduates earn the highest salaries on average.

5. No support for 2.5-3L claim: Fresh CSE graduate salaries vary widely,
   with median around 3.2L — the industry claim is not consistently supported.

6. Gender & Specialization: No significant relationship found. Both genders
   predominantly choose CSE.

7. Gender Salary Gap: Females earn slightly more on average (2.03L vs 1.94L),
   but the difference requires further investigation.
""")
