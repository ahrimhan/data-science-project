#%% [markdown]
# # 4. Machine Learning Models
#%% [markdown]
# **Table of Contents**
# 
# [Data Load](#Data-Load)  
# [Regression](#Regression)
# * [Validation for Regression](#Validation-for-Regression)
#     * Evaluation metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R$^{2}$
#     * Cross Validation: Train/Test Split, Leave One Out (LOO), K-fold CV
# * [Linear Regression](#Linear-Regression)
# * [Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regression](#Random-Forest-Regressor,-Gradient-Boosting-Regressor,-Support-Vector-Regression)
# * [Results of Regression](#Results-of-Regression)  
# 
# [Classification](#Classification)
# * [Validation for Classification](#Validation-for-Classification)
#     * Evaluation metrics: Accuracy, AUC, precision, recall, f1
#     * Cross Validation: Stratified K-Folds Cross Validation (imbalanced classes)
# * [Logistic Regression](#Logistic-Regression)
# * [Decision Tree](#Decision-Tree)
# * [Decision Tree with GridSearchCV for Parameter Tuning](#Decision-Tree-with-GridSearchCV-for-Parameter-Tuning)
# * [Random Forest Classifier with GridSearchCV](#Random-Forest-Classifier-with-GridSearchCV)
# * [k-Nearest Neighbors Classifier with GridSearchCV](#k-Nearest-Neighbors-Classifier-with-GridSearchCV)
# * [Results of Classification](#Results-of-Classification)  

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import plotly.plotly as py
import plotly.graph_objs as go
from collections import Counter
from statistics import stdev, mean
import operator 


#%%
#using the defined function in another notebooks
import nbimporter
from exploratory_data_analysis import getColumns


#%%
pd.set_option('display.max_columns', None)


#%%
#target_col = 'Percentage Standard Exceeded'
target_col = 'Percentage Standard Not Met'

#%% [markdown]
# ## Data Load
# 
# We loaded the school data for the independent variables. 
# We have selected the important features in the exploratory data analysis section.
# 
# The aim is to predict the inferior scores, percentage of the standard "NOT" met, of schools.
# Based on these prediction models, we can 1) identify the schools that need help and 2) obtain important features affecting the lower scores of schools. 

#%%
# %store -r final_data
# %store -r df_schools
# %store -r studentGroup_types
# %store -r attr_important_features
# %store -r attr_important_features_inter


#%%
pd.set_option('display.max_columns', None)


#%%
final_data = pd.read_csv('final_school_data.csv', sep='\t', encoding='utf-8')
final_data.drop("Unnamed: 0", axis=1, inplace=True)


#%%
df_schools = pd.read_csv('df_schools.csv', sep='\t', encoding='utf-8')
df_schools.drop("Unnamed: 0", axis=1, inplace=True)


#%%
studentGroup_types = final_data.groupby(["Category", "Student Groups"], as_index=False).mean()
studentGroup_types = studentGroup_types.loc[:, ["Category", "Student Groups"]]


#%%
attr_important_features = pd.read_csv('attr_important_features.csv', sep='\t', encoding='utf-8')
attr_important_features.drop("Unnamed: 0", axis=1, inplace=True)
attr_important_features = attr_important_features['feature'].tolist()


#%%
attr_important_features_inter = pd.read_csv('attr_important_features_inter.csv', sep='\t', encoding='utf-8')
attr_important_features_inter.drop("Unnamed: 0", axis=1, inplace=True)
attr_important_features_inter = attr_important_features_inter['feature_inter'].tolist()


#%%
def Diff(li1, li2): 
    list_diff = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return list_diff 

#features_diff = Diff(X.columns.tolist(), attr_important_features) 

#%% [markdown]
# ## Regression
# 
# Supervised Learning
# 
# In the regression, we use the `Target_Avg_Percentage Standard Not Met` variable as a target variable. 

#%%
#independent columns
#delete: Keywords

#original ver.
#X = df_schools.drop(getColumns(df_schools, ["English", "Mathematics", "Target", "Rank", "NeedHelp"], "OR"), axis=1)
#
X = df_schools.drop(getColumns(df_schools, ["English", "Mathematics", "All Students", "Target", "Rank", "NeedHelp"], "OR"), axis=1)

X_Num = df_schools.drop(getColumns(df_schools, ["Pct", "English", "Mathematics"], "OR"), axis=1)
X_Pct = df_schools.drop(getColumns(df_schools, ["Num", "English", "Mathematics"], "OR"), axis=1)
X_feat = df_schools[attr_important_features_inter]

X_all = X.copy()
X_all_Num = X_Num.copy()
X_all_Pct = X_Pct.copy()
# #Include
# X = X[getColumns(X, ["Avg"])]

#target column 
#y = df_schools.iloc[:,-1]
#round(0) for chi square analysis
target_var_name = "Target_Avg_"+target_col
#X = X.drop([target_var_name], axis=1)
X_feat = X_feat.drop([target_var_name], axis=1)
y = df_schools[target_var_name].round(0)



#delete: specific column name
X = X.drop(['School Code'], axis=1)
X_Num = X_Num.drop(['School Code'], axis=1)
X_Pct = X_Pct.drop(['School Code'], axis=1)

#Data
print("Target variable: ", target_var_name)
print("Number of instances: ", X.shape[0])
#y = y.astype('float64') # Your y is of type object, so sklearn cannot recognize its type. 
print("Independent variables (size:{})".format(X.columns.size))
print(X.columns)  #attr_important_features = X.columns
#attr_important_features_inter

#save
reg_name = []
reg_result = []

#%% [markdown]
# ### Validation for Regression
# 
# #### Evaluation metrics
# 
# https://towardsdatascience.com/how-to-select-the-right-evaluation-metric-for-machine-learning-models-part-2-regression-metrics-d4a1a9ba3d74
# 
# - Mean Absolute Error (MAE): MAE is the mean of the absolute value of the errors. 
# - Root Mean Squared Error (RMSE): RMSE is the square root of the mean of the squared errors
# - R$^{2}$: R$^{2}$ is the number that indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. Basically, R$^{2}$ represents how accurate our model is. R$^{2}$ shows how well terms (data points) fit a curve or line. Adjusted R$^{2}$ also indicates how well terms fit a curve or line, but adjusts for the number of terms in a model. 
# 
# 
# #### Cross Validation
# 
# We need to split the data into training and testing sets, fitted a regression model to the training data, made predictions based on this data and tested the predictions on the test data. 
# 
# However, the train/test split technique takes to one extreme, $K$ may be set to 1 such that a single train/test split is created to evaluate the model. Thus, the **train/test split technique is not stable in that it may not split the data randomly and the data can be selected only from specific groups.** This will result in overfitting.
# 
# The Leave One Out Cross Validation (LOOCV) takes to another extreme, $K$ may be set to the total number of observations in the dataset such that each observation is given a chance to be the held out of the dataset. This is called leave-one-out cross-validation, or LOOCV for short. However, **LOO requires quite a large computation time.**
# 
# **Therefore, we the cross validations: K-Fold.** This cross validation is enough and appropriate for our model prediction.
# 
# Reference:
# 
# Train/Test Split and Cross Validation in Python:
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6  
# https://scikit-learn.org/stable/modules/cross_validation.html
#%% [markdown]
# ### Linear Regression
#%% [markdown]
# #### Train/Test Split

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size = 0.3, random_state=5)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#%% [markdown]
# We provide the coefficients of the regression model. 
# We sorted the coefficients in a descending order of absolute values.

#%%
from sklearn.linear_model import LinearRegression

name = "Linear Regression"
cv_name = "Train and test split"
n_splits = 1

model = LinearRegression()
model.fit(X_train, y_train)

print("Model: {}, Cross Validation: {}, Number splits: {}".format(name, cv_name, n_splits))

#%% [markdown]
# * Coefficients for Linear Regression using Train/Test Split

#%%
coeff_df = pd.DataFrame((model.coef_).round(2), X.columns, columns=['Coefficient'])  
coeff_df.loc[coeff_df['Coefficient'].abs().sort_values(ascending=False).index]
#print('Estimated intercept coefficient: {}'.format(model.intercept_))
#print('Coefficients: {}'.format(model.coef_))

#%% [markdown]
# * Predicted Results for Linear Regression using Train/Test Split

#%%
y_pred = model.predict(X_test)
df_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_result.head()

#%% [markdown]
# * Model Validation Results for Linear Regression using Train/Test Split

#%%
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

score_rmse = []
score_mae = []
score_r2 = []

score_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
score_mae.append(mean_absolute_error(y_test, y_pred))
score_r2.append(r2_score(y_test, y_pred))

print("\n**Results**")
print("Model: {}, Cross Validation: {}, Number splits: {}".format(name, cv_name, n_splits))

print(f'Root Mean Squared Error(RMSE): {np.mean(score_rmse): 0.4f}, Mean Absolute Error(MAE): {np.mean(score_mae): 0.4f}, R^2 score: {np.mean(score_r2): 0.4f}')  

#save
reg_name.append(name + " with " + str(n_splits) + " folds " + cv_name)
reg_result.append(('%.4f' %np.mean(score_rmse), '%.4f' %np.mean(score_mae), '%.4f' %np.mean(score_r2)))

#%% [markdown]
# #### Leave One Out (LOO)
# 
# Please be cautious for getting a score of zero using cross val score when using the cross validation of Leave One Out.
# https://stackoverflow.com/questions/50605956/getting-a-score-of-zero-using-cross-val-score

#%%
from sklearn.model_selection import LeaveOneOut

scoring_reg = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

loo = LeaveOneOut()
n_splits = loo.get_n_splits(X) #all samples

name = "Linear Regression"
cv_name = "Leave One Out (LOO)"
model = LinearRegression()

scores = cross_validate(model, X, y, cv=loo, scoring=scoring_reg)

print("\n**Results**")
print("Model: {}, Cross Validation: {}, Number splits: {}".format(name, cv_name, n_splits))

rmse = (-1*mean(scores['test_neg_mean_squared_error']))**0.5
mae = -1*mean(scores['test_neg_mean_absolute_error'])
r2 = mean(scores['test_r2'])

print(f'Root Mean Squared Error(RMSE): {rmse: 0.4f}, Mean Absolute Error(MAE): {mae: 0.4f}')

#save
reg_name.append(name + " with " + str(n_splits) + " folds " + cv_name)
reg_result.append(('%.4f' %rmse, '%.4f' %mae, '%.4f' %r2))

#%% [markdown]
# * K-Fold Cross Validation
# 
# https://machinelearningmastery.com/k-fold-cross-validation/
# 
# cross_val_predict vs. cross_val_score:  
# https://stackoverflow.com/questions/43613443/difference-between-cross-val-score-and-cross-val-predict

#%%
from sklearn.model_selection import KFold

n_splits = 10
seed = 7

print("Cross Validation: {} {} {}{}".format(cv_name, "(K =", n_splits, ")"))
kfold = KFold(n_splits=n_splits, random_state=seed) # Define the split 


#%%
name = "Linear Regression"
cv_name = "K folds"
model = LinearRegression()

score_rmse = []
score_mae = []
score_r2 = []

print("\n**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Number splits: {}".format(name, cv_name, "(K =", n_splits, ")", n_splits))


for train_index, test_index in kfold.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    model.fit(X_train, y_train)    
    
    y_pred = model.predict(X_test)
    df_regression = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    score_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    score_mae.append(mean_absolute_error(y_test, y_pred))
    score_r2.append(r2_score(y_test, y_pred))

print(f'Root Mean Squared Error(RMSE): {np.mean(score_rmse): 0.4f}, Mean Absolute Error(MAE): {np.mean(score_mae): 0.4f}, R^2 score: {np.mean(score_r2): 0.4f}')    

#%% [markdown]
# We can use the `cross_validate` instead the above multiple tedious methods.

#%%
name = "Linear Regression"
cv_name = "K folds"
model = LinearRegression()

scores = cross_validate(model, X, y, cv=kfold, scoring=scoring_reg)

print("\n**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Number splits: {}".format(name, cv_name, "(K =", n_splits, ")", n_splits))

rmse = (-1*mean(scores['test_neg_mean_squared_error']))**0.5
mae = -1*mean(scores['test_neg_mean_absolute_error'])
r2 = mean(scores['test_r2'])

print(f'Root Mean Squared Error(RMSE): {rmse: 0.4f}, Mean Absolute Error(MAE): {mae: 0.4f}, R^2 score: {r2: 0.4f}')

#save
reg_name.append(name + " with " + str(n_splits) + " folds CV")
reg_result.append(('%.4f' %rmse, '%.4f' %mae, '%.4f' %r2))


#%%
y_pred = cross_val_predict(model, X, y, cv=kfold)
plt.scatter(y, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")


#%%
pd.DataFrame({'Actual': y, 'Prediction': y_pred}).head() #8768 rows

#%% [markdown]
# **sklearn vs. stats** 
# 
# In the above, we use the `scikit-learn` libraries. In `scikit-learn`, summary statistics must be done by hand. In contrast, computing summary statistics is done automatically in `statsmodels`.    
# 
# `statsmodels` has advantage of providing statistical summary (like t score, p-values) for all predictors and for intercept. The `scikit-learn` does not have this feature.  
# 
# The `scikit-learn` has advantage in terms of writing the equation. In `scikit-learn`, we just need to split the data to get features (X) and labels (y), and provide X and y as input. In short, `scikit-learn` pretty much just provides the the predicted values and the estimates of the regression coefficients.  
# In case of `statsmodels`, we need to explicitly write the expression.
# 
# https://datatofish.com/statsmodels-linear-regression/

#%%
# Import regression modules
import statsmodels.api as sm
from statsmodels.formula.api import OLS


#%%
# statsmodels works nicely with pandas dataframes
# The thing inside the "quotes" is called a formula, a bit on that below
X = sm.add_constant(X) # adding a constant

model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)

#%% [markdown]
# ### Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regression
# 
# - Random forest regressor:   
# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if `bootstrap=True` (default).
# 
# - Gradient Boosting for regression:  
# GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.
# 
# - Support Vector Regression:  
# The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to datasets with more than a couple of 10000 samples. 
# For large datasets consider using `sklearn.linear_model.LinearSVR` or `sklearn.linear_model.SGDRegressor` instead, possibly after a `sklearn.kernel_approximation.Nystroem` transformer.

#%%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

cv_name = "K folds"

models = {
 'Random Forest Regressor': RandomForestRegressor(n_estimators=50, random_state=0),
 'Gradient Boosting for Regression': GradientBoostingRegressor(max_depth=1, random_state=0),
 #'Support Vector Regression': SVR(kernel='linear') #too much complex with over 8700 samples
}

for name, model in models.items():
    scores = cross_validate(model, X, y, cv=kfold,
                            scoring=scoring_reg)
    print("\n**Results**")
    print("Model: {}, Cross Validation: {} {} {}{}, Number splits: {}".format(name, cv_name, "(K =", n_splits, ")", n_splits))

    rmse = (-1*mean(scores['test_neg_mean_squared_error']))**0.5
    mae = -1*mean(scores['test_neg_mean_absolute_error'])
    r2 = mean(scores['test_r2'])  
    print(f'Root Mean Squared Error(RMSE): {rmse: 0.4f}, Mean Absolute Error(MAE): {mae: 0.4f}, R^2 score: {r2: 0.4f}')
    
    reg_name.append(name + " with " + str(n_splits) + " folds CV")
    reg_result.append(('%.4f' %rmse, '%.4f' %mae, '%.4f' %r2))
    
#Results with LOO:  
#RFR RMSE:  0.5301, MAE:  0.3253  
#GBR RMSE:  2.2447, MAE:  1.6408 

#%% [markdown]
# ### Results of Regression

#%%
print("Model Accuracy Results:")
for res in reg_result:
    print(reg_name[reg_result.index(res)], ":", "RMSE: ", res[0], ", MAE: ", res[1], ", R^2: ", res[2])

#%% [markdown]
# ==> The Random Forest Regressor worked best with Root Mean Squared Error(RMSE) 10.7672, Mean Absolute Error(MAE) 7.6985, and R^2 0.6761.
#%% [markdown]
# ## Classification
#%% [markdown]
# We loaded the school data for the independent variables. 
# We use the `NeedHelp` variable as a target variable. 
# `NeedHelp` indicates that if a school needs a help (1) or not (0).   
# **We have labeled schools with more than 80\% of students who do not meet the standard as needing help (1).**
# 
# - `NeedHelp` (dependent or response variable)
#     - class 1: `Target_Avg_Percentage Standard Not Met` > 80
#     - class 0: Otherwise

#%%
#independent columns
target_col2='NeedHelp'

#ver 1.
# X = df_schools.drop(getColumns(df_schools, ["English", "Mathematics", "Target", "Rank"], "OR"), axis=1)
# X = X.drop(['School Code'], axis=1)

#ver.
X = df_schools.drop(getColumns(df_schools, ["English", "Mathematics", "All Students", "Target", "Rank"], "OR"), axis=1)
X = X.drop(['School Code'], axis=1)

#ver.
# X = df_schools[attr_important_features]
# X = X.drop(getColumns(X, ["Target", "Rank", "Pct"], "OR"), axis=1)

print("Target variable:", target_col2)
print("Number of instances")
print(target_col2,"== 0: ", len(X.loc[X[target_col2] == 0]))
print(target_col2,"== 1: ", len(X.loc[X[target_col2] == 1]))
pctOneTarget = (len(X.loc[X[target_col2]==1])) / (len(X.loc[X[target_col2]==0])) * 100
print("The percentage of target == 1 over target == 0: ", '%0.2f' % pctOneTarget , "%\n")



#Data
y = df_schools[target_col2]
X = X.drop([target_col2], axis=1)

print("Number of data: ", X.shape[0])
print("Independent variables (size:{})".format(X.columns.size))
print(X.columns) 

#Results
clf_name = []
clf_result = []
clf_best = []


#%%
#result save
def saveClfModelResultCV(results, name):
    print('Accuracy: {:.4f}'.format(mean(results['test_accuracy'])))
    print('Precision: {:.4f}'.format(mean(results['test_precision'])))
    print('Recall: {:.4f}'.format(mean(results['test_recall'])))
    print('f1-score: {:.4f}'.format(mean(results['test_f1'])))
    print('roc_auc_score: {:.4f}'.format(mean(results['test_roc_auc'])))

    print("**Weighted average scores**")
    print('Weighted Avg Precision : {:.4f}'.format(mean(results['test_precision_weighted'])))
    print('Weighted Avg Recall : {:.4f}'.format(mean(results['test_recall_weighted'])))
    print('Weighted Avg f1-score : {:.4f}'.format(mean(results['test_f1_weighted'])))

    #save start
    clf_name.append(name)
    clf_result.append("")    
    clf_best.append(("",
                    '%.4f' %mean(results['test_accuracy']),
                    '%.4f' %mean(results['test_roc_auc']),
                    '%.4f' %mean(results['test_precision_weighted']),
                    '%.4f' %mean(results['test_recall_weighted']),
                    '%.4f' %mean(results['test_f1_weighted'])))


#%%
def saveClfModelResultGridCV(clfObj, name):
    #GridSearchCV
    results = clfObj.cv_results_
    best_idx = clfObj.best_index_
    
    print("\nBest parameters: ", clfObj.best_params_)
    print("Best model: ", clfObj.best_estimator_)

    #clf_result[0]
    print("\nBest Accuracy: {:.4f}".format(results['mean_test_accuracy'][best_idx]))
    #clf_result[1]
    print('Best roc_auc_score: {:.4f}'.format(results['mean_test_roc_auc'][best_idx]))
    #print('Mean roc_auc_score: {:.4f}'.format(mean(results['mean_test_roc_auc'])))

    print("**Weighted average scores**")
    print("Weighted Avg Precision: {:.4f}".format(results['mean_test_precision_weighted'][best_idx]))
    print("Weighted Avg Recall: {:.4f}".format(results['mean_test_recall_weighted'][best_idx]))
    print("Weighted Avg f1-score: {:.4f}".format(results['mean_test_f1_weighted'][best_idx]))

    #save start
    clf_name.append(name)
    clf_result.append((results['mean_test_accuracy'],
                       results['mean_test_roc_auc'],
                       results['mean_test_precision_weighted'],
                       results['mean_test_recall_weighted'],
                       results['mean_test_f1_weighted']
                      ))
    
    clf_best.append((clfObj.best_params_,
                    '%.4f' %results['mean_test_accuracy'][best_idx],
                    '%.4f' %results['mean_test_roc_auc'][best_idx],
                    '%.4f' %results['mean_test_precision_weighted'][best_idx],
                    '%.4f' %results['mean_test_recall_weighted'][best_idx],
                    '%.4f' %results['mean_test_f1_weighted'][best_idx]))
                    #'%.4f' %mean(results['mean_test_roc_auc'])

#%% [markdown]
# ### Validation for Classification
#%% [markdown]
# ==> We can see we have an imbalanced class: 3.69\% of our dataset belong to the target class `NeedHelp`. 
# 
# To overcome the problem of the imbalanced classes, we need to deal with this imbalanccsced classes properly.
# 
# 1. We use the `weighted` option when calculating the precision, recall, and f1 scores to reflect the mass of the classes.   
# It calculates metrics for each label and finds their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall. https://www.programcreek.com/python/example/75177/sklearn.cross_validation.cross_val_score
# 
# 2. We used the `Stratified K-Folds Cross Validation`.  
# This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class. In short, the stratification will ensure that the percentages of each class in your entire data will be the same (or very close to) within each individual fold.  
# https://stackoverflow.com/questions/32615429/k-fold-stratified-cross-validation-with-imbalanced-classes
#%% [markdown]
# #### Evaluation Metrics
# 
# https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/
# 
# - Accuracy
# - precision, recall, f1
# - AUC
# 
# We also measure the accuracy, precision, recall and f1-score.  
# We also present the **Receiver Operating Characteristic (ROC)** curve and **Area Under the Curve (AUC)**.

#%%
# print("True Positive Rate (TPR): TP/Actual YES = TP/(TP+FN)")
# print("False Positive Rate (FPR): FP/Actual NO = FP/(FP+TN)")
# https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py

def getSetofMetrics(y_test, y_pred):
    cnf_matrix = confusion_matrix(y_test, y_pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    print("FP:", FP)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    print("FN:", FN)
    TP = np.diag(cnf_matrix)
    print("TP:", TP)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    print("TN:", TN)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Precision #PPV
    precision = TP/(TP+FP) #positive predictive value
    # Recall #TPR  #true positive rate
    recall = TP/(TP+FN) #sensitivity, hit rate
    # F1
    f1 = 2*((precision*recall)/(precision+recall))
    # false positive rate
    FPR = FP/(FP+TN) #Fall out  
    # false negative rate
    FNR = FN/(TP+FN)    
    # true negative rate
    TNR = TN/(TN+FP) #Specificity    
    # Negative predictive value
    NPV = TN/(TN+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    FP, FN, TP, TN, TNR, NPV, FDR, ACC
    return precision, recall, f1, FPR, FNR


#%%
pd.DataFrame(
    [['TN', 'FP'], ['FN', 'TP']],
    columns=['Predicted Not', 'Predicted'],
    index=['True Not', 'True']
)

#%% [markdown]
# ### Logistic Regression 
# 
# https://stackoverflow.com/questions/37367405/python-scikit-learn-cant-handle-mix-of-multiclass-and-continuous  
# https://towardsdatascience.com/how-to-build-a-gender-classifier-in-python-using-scikit-learn-13c7bb502f2e  
# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
#%% [markdown]
# #### Train/Test Split 

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #see classification metrics: https://scikit-learn.org/stable/modules/classes.html  
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size = 0.3, random_state=5)

name = "Logistic Regression"
cv_name = "Train and test split"
n_splits = 1

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n**Results**")
print("Model: {}, Cross Validation: {}, Number splits: {}".format(name, cv_name, n_splits))
print('Classification Report: \n', classification_report(y_test, y_pred), '\n')

print(f'Accurcy: {accuracy_score(y_test, y_pred): 0.4f}')
print(f'roc_auc_score: {roc_auc_score(y_test, y_pred): 0.4f}')

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print("**Weighted average scores**")
print('Precision : {:.4f}'.format(precision))
print('Recall    : {:.4f}'.format(recall))
print('F-score   : {:.4f}'.format(fscore))
#print('Support   : {}'.format(support))

#%% [markdown]
# #### Stratified K-Folds Cross Validation

#%%
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

def classification_report_with_as(y_true, y_pred):
    print(classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score


#%%
cv_name = "Stratified K-Folds CV"
n_splits = 5
seed = 7

print("Stratified K-Folds Cross Validation: {} {} {}{}".format(cv_name, "(K =", n_splits, ")"))
kfold = StratifiedKFold(n_splits=n_splits, random_state=seed) # Define the split 
#kfold = KFold(n_splits=n_splits, random_state=seed) # Define the split 

name = "Logistic Regression"
model = LogisticRegression(solver='liblinear')

score_acc = []
score_precision = []
score_recall = []
score_f1 = []
score_auc = []

score_precision2 = []
score_recall2 = []
score_f12 = []

k=0

for train_index, test_index in kfold.split(X,y):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    model.fit(X_train, y_train)    
    
    y_pred = model.predict(X_test)
    df_logistic = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
   
    try:
        k=k+1
        print("K =", k)
        
        score_acc.append(accuracy_score(y_test, y_pred))
        score_precision.append(precision_score(y_test, y_pred))
        score_recall.append(recall_score(y_test, y_pred))
        score_f1.append(f1_score(y_test, y_pred))
        score_auc.append(roc_auc_score(y_test, y_pred))
        
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        score_precision2.append(precision)
        score_recall2.append(recall)
        score_f12.append(fscore)

    except ValueError:
        pass
    
    print(classification_report(y_test, y_pred), '\n')

print("\n**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Number splits: {}".format(name, cv_name, "(K =", n_splits, ")", n_splits))


print(f'Accurcy: {np.mean(score_acc): 0.4f}')
print(f'Precision: {np.mean(score_precision): 0.4f}')
print(f'Recall: {np.mean(score_recall): 0.4f}')
print(f'f1-score: {np.mean(score_f1): 0.4f}')
print(f'roc_auc_score: {np.mean(score_auc): 0.4f}')

print("**Weighted average scores**")
print(f'Weighted Avg Precision: {np.mean(score_precision2): 0.4f}')
print(f'Weighted Avg Recall: {np.mean(score_recall2): 0.4f}')
print(f'Weighted Avg f1-score: {np.mean(score_f12): 0.4f}')

#%% [markdown]
# Here we use the `cross_validate` for automating this process.

#%%
scoring_clf = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
               'precision_weighted', 'recall_weighted', 'f1_weighted']

name = "Logistic Regression"
model = LogisticRegression(solver='liblinear')
cv_name = "Stratified K-Folds CV"
n_splits = 5
seed = 7
#clf_name.append(name + " with " + str(n_splits) + " folds" + cv_name_save)
save_name = name+" using Stratified "+str(n_splits)+"-Folds CV"

print("Stratified K-Folds Cross Validation: {} {} {}{}".format(cv_name, "(K =", n_splits, ")"))
kfold = StratifiedKFold(n_splits=n_splits, random_state=seed) # Define the split 


try:
    results = cross_validate(model, X, y, cv=kfold, scoring=scoring_clf)
                         #, error_score='raise-deprecating', return_estimator=True)
    #scores = cross_val_score(model, X, y, cv=kfold, scoring=precision_recall_fscore_support(y_test, y_pred, average='weighted'))

except ValueError:
    pass        

# log_acc = mean(results['test_accuracy'])
# log_f1 =  mean(results['test_f1'])
# log_precision =  mean(results['test_precision'])
# log_recall =  mean(results['test_recall'])
# log_precision_weighted =  mean(results['test_precision_weighted'])
# print(sorted(results.keys()))

# print("**Results**")
# print("Model: {}, Cross Validation: {} {} {}{}, Number splits: {}".format(name, cv_name, "(K =", n_splits, ")", n_splits))

# print(f'Accurcy: {log_acc: 0.4f}, precision: {log_precision: 0.4f}, recall: {log_recall: 0.4f}, f1: {log_f1: 0.4f}')


print("\n**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Number splits: {}".format(name, cv_name, "(K =", n_splits, ")", n_splits))

# print('Accuracy: {:.4f}'.format(mean(results['test_accuracy'])))
# print('Precision: {:.4f}'.format(mean(results['test_precision'])))
# print('Recall: {:.4f}'.format(mean(results['test_recall'])))
# print('f1-score: {:.4f}'.format(mean(results['test_f1'])))
# print('roc_auc_score: {:.4f}'.format(mean(results['test_roc_auc'])))
# print("**Weighted average scores**")
# print('Weighted Avg Precision : {:.4f}'.format(mean(results['test_precision_weighted'])))
# print('Weighted Avg Recall : {:.4f}'.format(mean(results['test_recall_weighted'])))
# print('Weighted Avg f1-score : {:.4f}'.format(mean(results['test_f1_weighted'])))

#save start
saveClfModelResultCV(results, save_name)

#%% [markdown]
# ### Decision Tree
# 
# The decision tree classifier iteratively divides the working area (plot) into subpart by identifying lines.
# There are three key terms related to decision tree classifiers:
# 
# **Criterion**
# 
# - Impurity:  
# Impurity is when we have a traces of one class division into other.
# 
# - Entropy:  
# Entropy is a degree of randomness of elements. In other words, it is a measure of impurity. It is the negative summation of probability times the log of the probability of item x.
# 
# - Information gain:  
# Information Gain (n) = Entropy(x) — ([weighted average] * entropy(children for feature))  
# At every stage, a decision tree selects the one that gives the best information gain. An information gain of 0 means the feature does not divide the working set at all.
# 
# **Optimizing Decision Tree Performance**  
# https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# 
# - criterion : optional (default=”gini”) or Choose attribute selection measure: This parameter allows us to use the different-different attribute selection measure. Supported criteria are “gini” for the Gini index and “entropy” for the information gain.
# 
# - splitter : string, optional (default=”best”) or Split Strategy: This parameter allows us to choose the split strategy. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
# 
# - max_depth : int or None, optional (default=None) or Maximum Depth of a Tree: The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting (Source).
#%% [markdown]
# **Train/Test Split for Decision Tree**

#%%
from sklearn.tree import DecisionTreeClassifier

#Decision Tree
name = "Decision Tree"
cv_name = "Train and test split"
n_splits = 1


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size = 0.3, random_state=5)

#criterion='entropy', default='gini'
dtree = DecisionTreeClassifier(random_state=seed)
dtree

#%% [markdown]
# * Model Fitting

#%%
dtree.fit(X_train, y_train)

#%% [markdown]
# * Feature Importances for Decision Tree

#%%
feature_importances = pd.DataFrame(dtree.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).abs().sort_values('importance', ascending=False)
feature_importances

#%% [markdown]
# * Classification Report

#%%
y_pred = dtree.predict(X_test)
print('\nClassification Report: \n', classification_report(y_test, y_pred), '\n')

#%% [markdown]
# * Confusion Matrix
# 
# https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

#%%
pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['[Predicted Not] '+target_col2, '[Predicted] '+target_col2],
    index=['[True Not] '+target_col2, '[True] '+target_col2]
)

#%% [markdown]
# * ROC and AUC
# 
# https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

#%%
def getROCAUC_crossval(clfObj, name, X_train, y_train, X_test, y_test):

    clfObj.fit(X_train, y_train)
    
    # y_pred = clfObj.predict(X_test)
    probs = clfObj.predict_proba(X_test) # returns probability estimates for test set 
    y_pred_prob = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred)  
    roc_auc = auc(fpr, tpr)
    
    plt.title(name + '\nReceiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc


#%%
def getROCAUC(clfObj, name, X_test, y_test):

    y_pred = clfObj.predict(X_test)
    
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.title(name + '\nReceiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc

#%% [markdown]
# * Probability Predictions  
# 
# Another type of prediction you may wish to make is the probability of the data instance belonging to each class.
# This is called a probability prediction where given a new instance, the model returns the probability for each outcome class as a value between 0 and 1.

#%%
def getROCAUCProb(clfObj, name, X_test, y_test):
    probs = clfObj.predict_proba(X_test) # returns probability estimates for test set 
    y_pred_prob = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.title(name + '\nReceiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc


#%%
result_auc = getROCAUCProb(dtree, name, X_test, y_test)

#%% [markdown]
# #### K-Fold Cross Validation for Decision Tree

#%%
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# import warnings
# warnings.filterwarnings('always') 

#Decision Tree
name = "Decision Tree"
cv_name = "Stratified K-Folds CV"
cv_name_save = "Stratified CV"
n_splits = 5
save_name = name+" using Stratified "+str(n_splits)+"-Folds CV"

#criterion='entropy', default='gini'
dtree = DecisionTreeClassifier(random_state=seed)

#results of arrays each of 5 elements (5 folds cv) 
results = cross_validate(dtree, X, y, cv=kfold, scoring=scoring_clf, return_estimator=True)

print("\n**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Number splits: {}".format(name, cv_name, "(K =", n_splits, ")", n_splits))

# print('Accuracy: {:.4f}'.format(mean(results['test_accuracy'])))
# print('Precision: {:.4f}'.format(mean(results['test_precision'])))
# print('Recall: {:.4f}'.format(mean(results['test_recall'])))
# print('f1-score: {:.4f}'.format(mean(results['test_f1'])))
# print('roc_auc_score: {:.4f}'.format(mean(results['test_roc_auc'])))
# print("**Weighted average scores**")
# print('Weighted Avg Precision : {:.4f}'.format(mean(results['test_precision_weighted'])))
# print('Weighted Avg Recall : {:.4f}'.format(mean(results['test_recall_weighted'])))
# print('Weighted Avg f1-score : {:.4f}'.format(mean(results['test_f1_weighted'])))

#save start
saveClfModelResultCV(results, save_name)

#print(results['estimator'])


#%%
result_auc = getROCAUC_crossval(dtree, name, X_train, y_train, X_test, y_test)

#%% [markdown]
# ### Decision Tree with GridSearchCV for Parameter Tuning
# 
# We first split the train and test data. In the train data, we use the k-fold cross validation for finding (tuning) the parameters. After the finding parameter process is finished, we use the remained test data to evaluate the model.
# 
# Please note that using the `GridSearchCV` for parameter tuning can be burden in time and computation.
# For example, for a model, if we consider 10-fold validation, 3 parameters in which one of each has 5 values, then the model needs to be run 1250 (= 5 * 5 * 5 * 10) times.
# 
# ![grid_search_cross_validation](grid_search_cross_validation.png)
# Image from https://scikit-learn.org/stable/modules/cross_validation.html

#%%
from sklearn.model_selection import GridSearchCV


scoring_clf_gridcv = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = {
    'max_depth': [50, 75, 100], #list(range(1,5))
    'min_samples_leaf': [1, 2, 4, 8, 10]
}

#5*3 = 15 cases (parameter differences)
#print(results['mean_test_roc_auc'])

#Decision Tree
name = "Decision Tree with GridSearchCV"
cv_name = "Stratified K-Folds CV"
cv_name_save = "Stratified CV"
dtree = DecisionTreeClassifier(random_state=seed)
n_splits = num_folds = 5
save_name = name
#+" using Stratified "+str(n_splits)+"-Folds CV"

# instantiate the grid
# one score (scoring='accuracy'), dtc_grid.cv_results_['mean_test_score']
# dtc_grid = GridSearchCV(estimator = dtc, param_grid = param_grid, scoring='accuracy', return_train_score=False,
#                      cv = num_folds, n_jobs = -1)

# need the standard metric - accuracy
dtc_grid = GridSearchCV(estimator = dtree, param_grid = param_grid,
                  scoring=scoring_clf_gridcv,  cv = num_folds, n_jobs = -1, refit='accuracy', return_train_score=True)

# instantiate and fit the grid
dtc_grid.fit(X_train, y_train)

print("**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Grid Search CV with parameters: {}".format(name, cv_name, "(K =", n_splits, ")", param_grid))

##############
#print(dtc_grid.cv_results_.keys())
print(pd.DataFrame(dtc_grid.cv_results_)[['mean_test_accuracy', 'std_test_accuracy', 'params']])

y_pred = dtc_grid.predict(X_test)
print('\nClassification Report: \n', classification_report(y_test, y_pred), '\n')

print('\nConfusion Matrix:')
target_col2='NeedHelp'

print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['[Predicted Not] '+target_col2, '[Predicted] '+target_col2],
    index=['[True Not] '+target_col2, '[True] '+target_col2]
))

# print("\nBest parameters: ", dtc_grid.best_params_)
# print("Best model: ", dtc_grid.best_estimator_)


# best_idx = dtc_grid.best_index_
# print("\nBest Accuracy: {:.4f}".format(results['mean_test_accuracy'][best_idx]))
# print('Best roc_auc_score: {:.4f}'.format(results['mean_test_roc_auc'][best_idx]))
# print('Mean roc_auc_score: {:.4f}'.format(mean(results['mean_test_roc_auc'])))

# print("**Weighted average scores**")
# print("Weighted Avg Precision: {:.4f}".format(results['mean_test_precision_weighted'][best_idx]))
# print("Weighted Avg Recall: {:.4f}".format(results['mean_test_recall_weighted'][best_idx]))
# print("Weighted Avg f1-score: {:.4f}".format(results['mean_test_f1_weighted'][best_idx]))


saveClfModelResultGridCV(dtc_grid, save_name)


#%%
result_auc = getROCAUCProb(dtc_grid, name, X_test, y_test)

#%% [markdown]
# * Feature Importances for Best Decision Tree found using GridSearchCV

#%%
feature_importances = pd.DataFrame(dtc_grid.best_estimator_.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).abs().sort_values('importance', ascending=False)
feature_importances

#%% [markdown]
# * Decision Tree Visualization

#%%
#https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

#class_names = ascending order
dot_data = StringIO()
export_graphviz(dtc_grid.best_estimator_, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=X_train.columns, class_names=['no_need_help','need_help'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
#graph.write_pdf("school.pdf")

#%% [markdown]
# ### Random Forest Classifier with GridSearchCV

#%%
from sklearn.ensemble import RandomForestClassifier

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth':[100, 150, 200],
    'min_samples_leaf':[1, 2, 4]
}

#Random Forest Classifier
name = "Random Forest Classifier with GridSearchCV"
cv_name = "Stratified K-Folds CV"
cv_name_save = "Stratified CV"
n_splits = num_folds = 5
save_name = name
#+" using Stratified "+str(n_splits)+"-Folds CV"

rfc = RandomForestClassifier()

# instantiate the grid
rfc_grid = GridSearchCV(estimator = rfc, param_grid = param_grid,
                  scoring=scoring_clf_gridcv,  cv = num_folds, n_jobs = -1, refit='accuracy', return_train_score=True)

# instantiate and fit the grid
rfc_grid.fit(X_train, y_train)


print("**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Grid Search CV with parameters: {}".format(name, cv_name, "(K =", n_splits, ")", param_grid))

print(pd.DataFrame(rfc_grid.cv_results_)[['mean_test_accuracy', 'std_test_accuracy', 'params']])

y_pred = rfc_grid.predict(X_test)
print('\nClassification Report: \n', classification_report(y_test, y_pred), '\n')

print('\nConfusion Matrix:')
target_col2='NeedHelp'

print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['[Predicted Not] '+target_col2, '[Predicted] '+target_col2],
    index=['[True Not] '+target_col2, '[True] '+target_col2]
))

# print("\nBest Accuracy: {:.4f}".format(rfc_grid.best_score_))
# print("Best parameters: ", rfc_grid.best_params_)
# print("Best model: ", rfc_grid.best_estimator_)

saveClfModelResultGridCV(rfc_grid, save_name)

#%% [markdown]
# * Feature Importnaces in Best Random Forest Model using GridSearchCV

#%%
feature_importances = pd.DataFrame(rfc_grid.best_estimator_.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).abs().sort_values('importance', ascending=False)
feature_importances


#%%
result_auc = getROCAUCProb(rfc_grid, name, X_test, y_test)

#%% [markdown]
# ### k-Nearest Neighbors Classifier with GridSearchCV

#%%
# from sklearn.preprocessing import StandardScaler

# # scaler = StandardScaler()
# # scaler.fit(X_train) #X_std = scaler.fit_transform(X_train)

# X_train_stdscal = scaler.transform(X_train)
# X_test_stdscal = scaler.transform(X_test)

#%% [markdown]
# * Without Scaler

#%%
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': list(range(1,31,2)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}


name = "K-Nearest Neighbor with GridSearchCV (No Scale)"
cv_name = "Stratified K-Folds CV"
cv_name_save = "Stratified CV"
_splits = num_folds = 5
save_name = name
#+" using Stratified "+str(n_splits)+"-Folds CV"

knn_noscale = KNeighborsClassifier()

# instantiate the grid
knn_grid_noscale = GridSearchCV(estimator = knn_noscale, param_grid = param_grid,
                  scoring=scoring_clf_gridcv,  cv = num_folds, n_jobs = -1, refit='accuracy', return_train_score=True)


# instantiate and fit the grid
knn_grid_noscale.fit(X_train, y_train)


print("**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Grid Search CV with parameters: {}".format(name, cv_name, "(K =", n_splits, ")", param_grid))

print(pd.DataFrame(knn_grid_noscale.cv_results_)[['mean_test_accuracy', 'std_test_accuracy', 'params']])

y_pred = knn_grid_noscale.predict(X_test)
print('\nClassification Report: \n', classification_report(y_test, y_pred), '\n')

print('\nConfusion Matrix:')
target_col2='NeedHelp'

print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['[Predicted Not] '+target_col2, '[Predicted] '+target_col2],
    index=['[True Not] '+target_col2, '[True] '+target_col2]
))

saveClfModelResultGridCV(knn_grid_noscale, save_name)


#%%
result_auc = getROCAUCProb(knn_grid_noscale, name, X_test, y_test)

#%% [markdown]
# * MinMaxScaler

#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#%%
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': list(range(1,31,2)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}


name = "K-Nearest Neighbor with GridSearchCV (Scaling)"
cv_name = "Stratified K-Folds CV"
cv_name_save = "Stratified CV"
_splits = num_folds = 5
save_name = name
#+" using Stratified "+str(n_splits)+"-Folds CV"

knn = KNeighborsClassifier()


# instantiate the grid
knn_grid = GridSearchCV(estimator = knn, param_grid = param_grid,
                  scoring=scoring_clf_gridcv,  cv = num_folds, n_jobs = -1, refit='accuracy', return_train_score=True)


# instantiate and fit the grid
knn_grid.fit(X_train, y_train)


print("**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Grid Search CV with parameters: {}".format(name, cv_name, "(K =", n_splits, ")", param_grid))

print(pd.DataFrame(knn_grid.cv_results_)[['mean_test_accuracy', 'std_test_accuracy', 'params']])

y_pred = knn_grid.predict(X_test)
print('\nClassification Report: \n', classification_report(y_test, y_pred), '\n')

print('\nConfusion Matrix:')
target_col2='NeedHelp'

print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['[Predicted Not] '+target_col2, '[Predicted] '+target_col2],
    index=['[True Not] '+target_col2, '[True] '+target_col2]
))

saveClfModelResultGridCV(knn_grid, save_name)


#%%
result_auc = getROCAUCProb(knn_grid, name, X_test, y_test)

#%% [markdown]
# * Comparison (Scale vs. No Scale)

#%%
print("**Boxplot of Accuracy Comparison for Scaling and No Scaling data for K-Nearest Neighbor Models**")

fig = plt.figure(figsize=(20,10))
fig.suptitle('Accuracy Comparison: Scaling and No Scaling data for K-Nearest Neighbor Models')
ax = fig.add_subplot(111)
#x[1] : accuracy
plt.boxplot([x[1] for x in clf_result[2:4] if x != ''])
clf_gs_name = [name for name in clf_name if "K-Nearest" in name]
ax.set_xticklabels(clf_gs_name)
plt.show()

#%% [markdown]
# ### SVC (Support Vector Classifier) with GridSearchCV
# 
# https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02

#%%
# from sklearn.svm import SVC

# param_grid = {
#     #'C': [0.001, 0.01, 0.1, 1, 10],
#     'C': [0.1, 1, 10, 100, 1000],
#     #'gamma': [0.001, 0.01, 0.1, 1]
#     'gamma': [0.1, 1, 10, 100]
# }


# #Support Vector Classifier
# name = "Support Vector Classifier with GridSearchCV"
# cv_name = "Stratified K-Folds CV"
# cv_name_save = "Stratified CV"
# _splits = num_folds = 5
# save_name = name+" using Stratified "+str(n_splits)+"-Folds CV"

# svc = SVC(kernel='linear', class_weight='balanced')

# # instantiate the grid
# svc_grid = GridSearchCV(estimator = svc, param_grid = param_grid,
#                   scoring=scoring_clf_gridcv,  cv = num_folds, n_jobs = -1, refit='accuracy', return_train_score=True)


# # instantiate and fit the grid
# svc_grid.fit(X_train, y_train)


# print("**Results**")
# print("Model: {}, Cross Validation: {} {} {}{}, Grid Search CV with parameters: {}".format(name, cv_name, "(K =", n_splits, ")", param_grid))

# print(pd.DataFrame(svc_grid.cv_results_)[['mean_test_accuracy', 'std_test_accuracy', 'params']])


# y_pred = svc_grid.predict(X_test)
# print('\nClassification Report: \n', classification_report(y_test, y_pred), '\n')

# print('\nConfusion Matrix:')
# target_col2='NeedHelp'

# print(pd.DataFrame(
#     confusion_matrix(y_test, y_pred),
#     columns=['[Predicted Not] '+target_col2, '[Predicted] '+target_col2],
#     index=['[True Not] '+target_col2, '[True] '+target_col2]
# ))

# saveClfModelResultGridCV(svc_grid, save_name)


#%%
#result_auc = getROCAUCProb(svc_grid, name, X_test, y_test)

#%% [markdown]
# ### Gradient Boosting
# 
# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae

#%%
# from sklearn.ensemble import GradientBoostingClassifier

# param_grid = {
#     'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
#     'max_depths' = np.linspace(1, 32, 32, endpoint=True),
#     'min_samples_splits' = np.linspace(0.1, 1.0, 10, endpoint=True)
#     'min_samples_leafs' = np.linspace(0.1, 0.5, 5, endpoint=True)
#     'max_features' = list(range(1, X_train.shape[1]))
# }

# name = "Gradient Boosting Classifier with GridSearchCV"
# cv_name = "Stratified K-Folds CV"
# cv_name_save = "Stratified CV"
# _splits = num_folds = 5
# save_name = name+" using Stratified "+str(n_splits)+"-Folds CV"

# gbc = GradientBoostingClassifier()


# # instantiate the grid
# gbc_grid = GridSearchCV(estimator = gbc, param_grid = param_grid,
#                   scoring=scoring_clf_gridcv,  cv = num_folds, n_jobs = -1, refit='accuracy', return_train_score=True)


# # instantiate and fit the grid
# gbc_grid.fit(X_train, y_train)


# print("**Results**")
# print("Model: {}, Cross Validation: {} {} {}{}, Grid Search CV with parameters: {}".format(name, cv_name, "(K =", n_splits, ")", param_grid))

# print(pd.DataFrame(gbc_grid.cv_results_)[['mean_test_accuracy', 'std_test_accuracy', 'params']])

# y_pred = gbc_grid.predict(X_test)
# print('\nClassification Report: \n', classification_report(y_test, y_pred), '\n')

# print('\nConfusion Matrix:')
# target_col2='NeedHelp'

# print(pd.DataFrame(
#     confusion_matrix(y_test, y_pred),
#     columns=['[Predicted Not] '+target_col2, '[Predicted] '+target_col2],
#     index=['[True Not] '+target_col2, '[True] '+target_col2]
# ))

# saveClfModelResultGridCV(gbc_grid, save_name)

#%% [markdown]
# ### Results of Classification

#%%
clf_gs_name


#%%
print("**Boxplot of Accuracy Comparison for GridSearch CV Models**")
fig = plt.figure(figsize=(20,10))
fig.suptitle('Accuracy Comparison: Models using GridSearchCV')
ax = fig.add_subplot(111)
#x[1] : accuracy
plt.boxplot([x[1] for x in clf_result if x != ''])
clf_gs_name = [name for name in clf_name if "GridSearchCV" in name]
ax.set_xticklabels(clf_gs_name)
plt.show()


#%%
print("**Best Parameters and Best Scores**")
df_result = pd.DataFrame(clf_best, columns=['best_param', 'mean_accuracy', 'mean_roc_auc', 'weighted_average_precision', 'weighted_average_recall', 'weighted_average_f1'])
df_result.insert(loc=0, column='model name', value=clf_name)
df_result

#%% [markdown]
# ==> For conclusion, the **Random Forest Classifier worked best**.  
# 
# After scaling the train data using `MinMaxScaler`, the scores for the **best K-Nearest Neighbor** signaficantly improved.
#%% [markdown]
# ## Clustering - TBD
# 
# Unsupervised Learning
#%% [markdown]
# ## References
# 
# 
# Predicting School Performance with Census Income Data  
# https://towardsdatascience.com/predicting-school-performance-with-census-income-data-ad3d8792ac97
# 
# Gridsearch CV: https://github.com/justmarkham/scikit-learn-videos  
# Grid search user guide: https://scikit-learn.org/stable/modules/grid_search.html  
# GridSearchCV documentation:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html  
# RandomizedSearchCV documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html  
# Comparing randomized search and grid search: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html  
# Random Search for Hyper-Parameter Optimization: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf  
# Model result comparison: https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/  
# Regression:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html  
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f  
# k-Neighbors Classifier with GridSearchCV Basics:
# https://medium.com/@erikgreenj/k-neighbors-classifier-with-gridsearchcv-basics-3c445ddeb657
# 
# Dealing with imbalanced data for classification:
# https://www.kaggle.com/tboyle10/methods-for-dealing-with-imbalanced-data  
# k-fold stratified cross-validation with imbalanced classes:  
# https://stackoverflow.com/questions/32615429/k-fold-stratified-cross-validation-with-imbalanced-classes
# 
# Different result with roc_auc_score() and auc():
# https://stackoverflow.com/questions/31159157/different-result-with-roc-auc-score-and-auc  
# 
# Accessing a value in a tuple that is in a list
# https://stackoverflow.com/questions/4800811/accessing-a-value-in-a-tuple-that-is-in-a-list
# In Depth: Parameter tuning for SVC  
# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
# Imbalanced Classes In SVM  
# https://chrisalbon.com/machine_learning/support_vector_machines/imbalanced_classes_in_svm/
# 
# Imbalanced Data XGBoost Tunning  
# https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
# 
# How To Prepare Your Data For Machine Learning in Python with Scikit-Learn  
# https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/
# 
# Running Random Forests? Inspect the feature importances with this code.
# https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e
#%% [markdown]
# **Processing using Pipeline**

#%%
# # Create first pipeline for base without reducing features.

# pipe = Pipeline([('classifier' , RandomForestClassifier())])
# # pipe = Pipeline([('classifier', RandomForestClassifier())])

# # Create param grid.

# param_grid = [
#     {'classifier' : [LogisticRegression()],
#      'classifier__penalty' : ['l1', 'l2'],
#     'classifier__C' : np.logspace(-4, 4, 20),
#     'classifier__solver' : ['liblinear']},
#     {'classifier' : [RandomForestClassifier()],
#     'classifier__n_estimators' : list(range(10,101,10)),
#     'classifier__max_features' : list(range(6,32,5))}
# ]

# # Create grid search object

# clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

# # Fit on data

# best_clf = clf.fit(X_train, y_train)


