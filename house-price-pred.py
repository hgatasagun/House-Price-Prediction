##################################################################
# # House Price Prediction Model
#################################################################

## Problem Statement
# Predicting house prices for different types of houses based on their features using machine learning.

## Dataset Overview
# The dataset contains information on 79 explanatory variables and sale prices of houses located in Ames, Iowa.
# The dataset is from a Kaggle competition, and it's divided into two separate CSV files: train and test.
# The train dataset includes the sale prices, while the test dataset has the sale prices left blank, requiring
# you to predict them.


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import combinations
from scipy.stats import chi2_contingency
from scipy.stats import skew

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import warnings
warnings.simplefilter("ignore")

################################################
# TASK 1 - EXPLORATORY DATA ANALYSIS (EDA)
################################################

# Step 1: Read and Combine Train and Test Datasets
##################################################
data1 = pd.read_csv('/Users/handeatasagun/PycharmProjects/Case studies-Miuul/datasets/train_ev.csv')
data2 = pd.read_csv('/Users/handeatasagun/PycharmProjects/Case studies-Miuul/datasets/test_ev.csv')

df = pd.concat([data1, data2], ignore_index=True)



# Convert Column Names to Lowercase
###########################################
df.columns = [col.lower() for col in df.columns]


# Check DataFrame Information
#############################
def check_df(dataframe, head=5):
    print('################# Columns ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)


check_df(df)



# Capture Numerical and Categorical Variables
#############################################
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Perform Necessary Data Adjustments
####################################
df['mssubclass'] = df['mssubclass'].astype(object)

df['yrsold'] = df['yrsold'].astype(int)

df = df.drop('id',axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    print(col, df[col].unique())

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Explore the Distribution of Numerical and Categorical Variables
##################################################################

# Categorical variables:
########################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


# Dropping Highly Imbalanced Categorical Variables
##################################################
for col in cat_cols:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)

columns_to_drop = []
for col in cat_cols:
    ratio = df[col].value_counts(normalize=True).max()
    if ratio > 0.9:
        columns_to_drop.append(col)

df = df.drop(columns=columns_to_drop)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Numerical variables:
######################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot = True)



# Analyze Categorical Variables in Relation to the Target Variable
###################################################################

def target_summary_with_cat(dataframe, target, categorical_col):
   print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, 'saleprice', col)



# Analyzse Outliers
###########################################


# Outlier thresholds calculation
################################
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

sns.boxplot(data=df, palette="Set3", showmeans=True)


# Define a function to check for outliers in a column
#####################################################
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))


# Define a function to identify and display outliers
####################################################
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    outlier_df = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]

    if outlier_df.shape[0] > 10:
        print(outlier_df.head())
    else:
        print(outlier_df)

    if index:
        outlier_index = outlier_df.index
        return outlier_index

    return outlier_df.shape[0]

for col in num_cols:
    print(col, grab_outliers(df, col))



# Analyse Missing Values
########################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

#               n_miss  ratio
# poolqc          2909 99.660
# miscfeature     2814 96.400
# alley           2721 93.220
# fence           2348 80.440
# saleprice       1459 49.980 # target
# fireplacequ     1420 48.650

df = df.drop(columns=['poolqc', 'alley', 'fence', 'fireplacequ'])

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Define a function to fill missing values with mean (numerical) and mode (categorical)
#######################################################################################
def fill_missing(df, target_col):
    filled_df = df.copy()

    for column in filled_df.columns:
        if column != target_col and filled_df[column].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(filled_df[column]):
                column_mean = filled_df[column].mean()
                filled_df[column].fillna(column_mean, inplace=True)
            elif filled_df[column].dtype == 'O':
                column_mode = filled_df[column].mode()[0]
                filled_df[column].fillna(column_mode, inplace=True)

    return filled_df

df = fill_missing(df, target_col='saleprice')

df.isnull().sum()
df.head()



################################################
# TASK 2: FEATURE ENGINEERING
################################################

# Label encoder for ordinal variables
#####################################
def label_encoder(dataframe, col):
    labelencoder = LabelEncoder()
    for c in col:
      dataframe[c] = labelencoder.fit_transform(dataframe[c])
    return dataframe


le_col = ['lotshape', 'overallqual', 'overallcond', 'exterqual', 'extercond',
       'bsmtqual', 'bsmtexposure', 'bsmtfintype1', 'bsmtfintype2',
       'heatingqc', 'kitchenqual']

df = label_encoder(df, le_col)

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Check and Handle High Correlations Among Numerical Variables
####################3#########################################
def high_correlated_cols(dataframe, corr_th=0.7):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))

    correlated_pairs = []

    for col in upper_triangle_matrix.columns:
        high_corr_cols = upper_triangle_matrix.index[upper_triangle_matrix[col] > corr_th].tolist()
        for high_corr_col in high_corr_cols:
            correlation = corr.loc[col, high_corr_col]
            correlated_pairs.append((col, high_corr_col, correlation))

    return correlated_pairs


correlated_pairs = high_correlated_cols(df, corr_th=0.7)

for pair in correlated_pairs:
    print(f"High Correlation: {pair[0]} - {pair[1]} (Correlation Coefficient: {pair[2]:.3f})")

df = df.drop(columns=['1stflrsf', 'garageyrblt', 'garagecars', 'totrmsabvgrd', 'bsmtfintype2'])

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Check and Handle High Correlations Among Categorical Variables
################################################################

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2) / (n-1)
    k_corr = k - ((k-1)**2) / (n-1)
    cramers_v = np.sqrt(phi2corr / min((k_corr-1), (r_corr-1)))
    return cramers_v


cramer_threshold = 0.7

high_correlations = []


for col1, col2 in combinations(cat_cols, 2):
    cramer_value = cramers_v(df[col1], df[col2])
    if cramer_value > cramer_threshold:
        high_correlations.append((col1, col2, cramer_value))

for col1, col2, cramer_value in high_correlations:
    print(f"High Correlation: {col1} - {col2} (Cramér's V Value: {cramer_value:.3f})")

df = df.drop(columns=['mssubclass', 'exterior1st'])

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Create New Features
#####################
significant_interactions = {}

for col in cat_cols:
    target_summary = df.groupby(col)['saleprice'].mean()
    if len(target_summary) > 1 and col != 'saleprice':
        max_diff = target_summary.max() - target_summary.min()
        if max_diff > 220000:
            significant_interactions[col] = target_summary

for feature, summary in significant_interactions.items():
    print(f"Feature: {feature}")
    print(summary)
    print("=" * 30)

interaction_columns = []

for i, col1 in enumerate(significant_interactions):
    for j, col2 in enumerate(significant_interactions):
        if j > i:
            new_col_name = f'{col1}_{col2}'
            interaction_columns.append(new_col_name)
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                df[new_col_name] = df[col1] * df[col2]
            else:
                df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)


cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.head()

df['exterqual_kitchenqual'].dtype

df['_overallqual_x_grlivarea'] = df['overallqual'] * df['grlivarea']

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Rare Encoder
###############

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "saleprice", cat_cols)

df = df.drop(columns=['3ssnporch', 'poolarea', 'miscval', 'lowqualfinsf'])

cat_cols, num_cols, cat_but_car = grab_col_names(df)



def rare_encoder(dataframe, rare_perc, col=None):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        print(f"Column: {col}")
        print(tmp)
        print("=" * 30)

    return temp_df

df = rare_encoder(df, 0.05)

df.head()

for col in cat_cols:
    print(col, df[col].unique())

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Label Encoder - Binary Columns
#################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# One-hot Encoder
#################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if (10 >= df[col].nunique() > 2) & (col not in le_col) and (col not in binary_cols)]

df = one_hot_encoder(df, ohe_cols)

df.head()

df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df_ = df.copy()



# Check Skewness of Numerical Columns and Apply Log Transformation
##################################################################
for col in num_cols:
    if abs(skew(df[col])) > 0.8:
        print(f"Column: {col}, Skewness Value: {skew(df[col])}")

for col in num_cols:
    if abs(skew(df[col])) > 0.8:
        df[col] = np.log1p(df[col])

df.head()


# Standardization
##################
scaler = StandardScaler()

num_cols = [col for col in df.columns if col != 'saleprice']

df[num_cols] = scaler.fit_transform(df[num_cols])


#######################################
# TASK 3 - MODELLING
#######################################

# Splitting Train and Test Data
################################

# For Linear models and KNN
###########################
test_df = df.iloc[-1459:]
test_df.shape

train_df = df.iloc[:-1459]
train_df.shape


# Linear regression and KNN - without log transformation in target variable
###########################################################################

def evaluate_model(model, X_train, y_train):
    pipeline = Pipeline([('model', model)])

    cv_results = cross_validate(model,
                                X_train, y_train,
                                cv=5,
                                scoring='neg_mean_squared_error')

    rmse = np.sqrt(-cv_results['test_score'])
    print("Average Cross-Validation RMSE:", np.mean(rmse))

    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)

    r2_train = r2_score(y_train, y_pred_train)

    return r2_train, rmse


X_train = train_df.drop("saleprice", axis=1)
y_train = train_df["saleprice"]
X_test = test_df.drop("saleprice", axis=1)

models_1 = [
    ("Linear Regression", LinearRegression()),
    ("Lasso Regression", Lasso(alpha=0.01)),
    ("Ridge Regression", Ridge(alpha=0.01)),
    ("ElasticNet Regression", ElasticNet(alpha=0.01, l1_ratio=0.5)),
    ("K-Nearest Neighbors", KNeighborsRegressor())
]

results_df_1 = pd.DataFrame(columns=['Model', 'R2 Score (Train)', 'RMSE'])

for model_name, model in models_1:
    r2_train, rmse_train = evaluate_model(model, X_train, y_train)
    results_df_1 = results_df_1.append({'Model': model_name, 'R2 Score (Train)': r2_train, 'RMSE': np.mean(rmse_train)}, ignore_index=True)

print(results_df_1)
#                    Model  R2 Score (Train)      RMSE
# 0      Linear Regression             0.874 39355.463
# 1       Lasso Regression             0.874 39348.289
# 2       Ridge Regression             0.874 39325.826
# 3  ElasticNet Regression             0.873 33149.291
# 4    K-Nearest Neighbors             0.858 37338.169


# Linear regression and KNN - with log transformation in target variable
########################################################################

def evaluate_model_with_log_transform(model, X_train, y_train):
    pipeline = Pipeline([('model', model)])

    y_train_log = np.log(y_train)

    cv_results = cross_validate(model,
                                X_train, y_train_log,
                                cv=5,
                                scoring='neg_mean_squared_error')

    rmse = np.sqrt(-cv_results['test_score'])
    print("Average Cross-Validation RMSE (Log-Transformed):", np.mean(rmse))

    pipeline.fit(X_train, y_train_log)

    y_pred_train_log = pipeline.predict(X_train)

    y_pred_train = np.exp(y_pred_train_log)
    y_train_exp = np.exp(y_train_log)

    r2_train = r2_score(y_train_exp, y_pred_train)

    rmse_original_scale = np.sqrt(mean_squared_error(y_train_exp, y_pred_train))
    print("R2 Score (Original Scale):", r2_train)
    print("RMSE (Original Scale):", rmse_original_scale)

    return r2_train, rmse_original_scale


results_df3 = pd.DataFrame(columns=['Model', 'R2 Score (Train)', 'RMSE'])

for model_name, model in models_1:
    r2_train, rmse_train = evaluate_model_with_log_transform(model, X_train, y_train)
    rmse_mean = np.mean(rmse_train)
    results_df3 = results_df3.append({'Model': model_name, 'R2 Score (Train)': r2_train, 'RMSE': rmse_mean}, ignore_index=True)

print(results_df3)

#                    Model  R2 Score (Train)      RMSE
# 0      Linear Regression             0.897 25256.535
# 1       Lasso Regression             0.858 29682.297
# 2       Ridge Regression             0.897 25256.495
# 3  ElasticNet Regression             0.877 27674.335
# 4    K-Nearest Neighbors             0.851 30387.149

