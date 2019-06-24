#%% [markdown]
# # 3. Exploratory Data Analysis
#%% [markdown]
# **Table of Contents**
# 
# [Data into School Level Instances for Prediction](#Data-into-School-Level-Instances-for-Prediction)  
# * [Process](#Process)
# * [Imputing Missing Values](#Imputing-Missing-Values)
# 
# [Statistical Hypothesis Testing](#Statistical-Hypothesis-Testing)
# * [T-Test for means of two independent samples](#T-Test-for-means-of-two-independent-samples)  
# 
# [Correlation Tests](#Correlation-Tests)
# * [Matrix with Heatmap](#Matrix-with-Heatmap)
# * [Pearson’s Correlation Coefficient](#Pearson’s-Correlation-Coefficient)
# * [Spearman's Rank Correlation](#Spearman's-Rank-Correlation)
# 
# [Feature Selection](#Feature-Selection)  
# * [Univariate Selection](#Univariate-Selection)
# * [Feature Importance](#Feature-Importance)
# 
# [Further Analysis for Recommendations](#Further-Analysis-for-Recommendations)
# 
# [Variables for Modeling](#Variables-for-Modeling)
# * [Decision for Variables](#Decision-for-Variables)
# * [Independent and Dependent Variables](#Independent-and-Dependent-Variables)

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
class Exploratory_data_analysis():
    def __init__(self):
        print("Imported functions from exploratory_data_analysis notebook")

#%% [markdown]
# We first set a target variable, e.g., 'Percentage Standard Exceeded' or 'Percentage Standard Not Met', to be investigated. We are interested in the group of students whose performance achievements are exceeded or too inferior. By knowing the characteristics affecting those groups, we can make a score prediction and suggest recommendations later.

#%%
target_col = 'Percentage Standard Exceeded'
#target_col = 'Percentage Standard Not Met'


#%%
#global variables
ops = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le, '=': operator.eq}
attr_avg_score = ['Target_Avg_Percentage Standard Exceeded', 'Target_Avg_Percentage Standard Met', 'Target_Avg_Percentage Standard Nearly Met', 'Target_Avg_Percentage Standard Not Met']


#%%
pd.set_option('display.max_columns', None)


#%%
df = pd.read_csv('final_school_data.csv', sep='\t', encoding='utf-8')
final_data = df
final_data.drop("Unnamed: 0", axis=1, inplace=True)


#%%
final_data.head()


#%%
# df = final_data.loc[(final_data['School Name'] == 'Eastwood Elementary') & 
#              (final_data['District Name'] == 'Irvine Unified') & 
#              (final_data['County Name'] == 'Orange'), :]

df = final_data.copy()

#%% [markdown]
# ## Data into School Level Instances for Prediction
#%% [markdown]
# ### Process
# 
# For predicting school scores, we need to focus on the school-level instances.  
# Therefore, we transform and select data for each school.  
# We also need to derive new variables.
#%% [markdown]
# **1. Student Number and Percentage**

#%%
df_studentNum = pd.pivot_table(df, index=['School Code', 'Category', 'Student Groups', 'Test Id'], values=['Students with Scores'])
df_studentNum.reset_index(inplace=True)
df_studentNum.head()


#%%
#data.pivot_table(values, index, columns) vs. pandas.pivot_table(data, values, index)
df_studentNum_pivot_table = df_studentNum.pivot_table(values='Students with Scores', index='School Code', columns=['Category', 'Student Groups', 'Test Id'], aggfunc='mean', fill_value=0)
df_studentNum_pivot_table.head()


#%%
# concatenate all columns names
def concat_col_names(pivot_table, pre="", post=""):
    cols = pivot_table.columns.values.tolist()
    col_list=[]
    for c in cols:
        col_list.append(pre+"_".join(c)+post)
    pivot_table.columns = col_list       
    pivot_table.fillna(0)


#%%
concat_col_names(df_studentNum_pivot_table, pre="Num_")   
df_studentNum_pivot_table.head()


#%%
#Calculate the percentage variables
for col in df_studentNum_pivot_table.columns:
    if "Num_" in col and col.endswith("English"):
        #Handling division by zero in Pandas calculations
        pct = (df_studentNum_pivot_table[col]/df_studentNum_pivot_table["Num_All Students_All Students_English"]).replace(np.inf, 0)*100
        df_studentNum_pivot_table["Pct_"+col.strip("Num_")] = pct.round(2)
    elif "Num_" in col and col.endswith("Mathematics"):
        #Handling division by zero in Pandas calculations
        pct = (df_studentNum_pivot_table[col]/df_studentNum_pivot_table["Num_All Students_All Students_Mathematics"]).replace(np.inf, 0)*100
        df_studentNum_pivot_table["Pct_"+col.strip("Num_")] = pct.round(2)
        
df_studentNum_pivot_table.head()

#%% [markdown]
# We also calculate the average of both English and Mathematics for number and percentage of students.

#%%
#df_studentNum_avg = df_studentNum.groupby(['School Code', 'Category', 'Student Groups'], as_index=False).mean()
df_studentNum_pivot_table_avg = df_studentNum.pivot_table(values='Students with Scores', index='School Code', columns=['Category', 'Student Groups'], aggfunc='mean', fill_value=0)
concat_col_names(df_studentNum_pivot_table_avg, pre="Num_Avg_") 

#Calculate the percentage variables
for col in df_studentNum_pivot_table_avg.columns:
    if "Num_" in col:
        #Handling division by zero in Pandas calculations
        pct = (df_studentNum_pivot_table_avg[col]/df_studentNum_pivot_table_avg["Num_Avg_All Students_All Students"]).replace(np.inf, 0)*100
        df_studentNum_pivot_table_avg["Pct_"+col.strip("Num_")] = pct.round(2)
        
#df_studentNum_pivot_table_avg.head()

#%% [markdown]
# We add new variables by combining Asian and Whites as well as Hispanic and Black students in the Ethnicity.  
# Each group lies and shows very similar pattern in the scores, such 'Percentage Standard Exceeded' or 'Percentage Standard Not Met', so we expect these merging can reduce the dimensionality or tell new insights.

#%%
#new variables (Asian+Whiate, Hispanic+Black)
# df_studentNum_pivot_table["Pct_Multi_Ethnicity_Asian+White_English"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_Asian_English"]+x["Pct_Ethnicity_White_English"], axis=1)
# df_studentNum_pivot_table["Pct_Multi_Ethnicity_Asian+White_Mathematics"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_Asian_Mathematics"]+x["Pct_Ethnicity_White_Mathematics"], axis=1)
# df_studentNum_pivot_table["Pct_Avg_Multi_Ethnicity_Asian+White"] = df_studentNum_pivot_table.apply(lambda x: (x["Pct_Ethnicity_Asian_English"]+x["Pct_Ethnicity_Asian_Mathematics"])/2 + (x["Pct_Ethnicity_White_English"]+x["Pct_Ethnicity_White_Mathematics"])/2, axis=1)

# df_studentNum_pivot_table["Pct_Multi_Ethnicity_Hispanic+Black_English"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_Hispanic or Latino_English"]+x["Pct_Ethnicity_Black or African American_English"], axis=1)
# df_studentNum_pivot_table["Pct_Multi_Ethnicity_Hispanic+Black_Mathematics"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_Hispanic or Latino_Mathematics"]+x["Pct_Ethnicity_Black or African American_Mathematics"], axis=1)
# df_studentNum_pivot_table["Pct_Avg_Multi_Ethnicity_Hispanic+Black"] = df_studentNum_pivot_table.apply(lambda x: (x["Pct_Ethnicity_Hispanic or Latino_English"]+x["Pct_Ethnicity_Hispanic or Latino_Mathematics"])/2 + (x["Pct_Ethnicity_Black or African American_English"]+x["Pct_Ethnicity_Black or African American_Mathematics"])/2, axis=1)


#%%
#new variables (Asian+Whiate, Hispanic+Black)
ethinicity_combi_cols = [['Asian', 'White'], ['Hispanic or Latino', 'Black or African American']]

df_studentNum_pivot_table["Pct_Multi_Ethnicity_Asian+White_English"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_"+ethinicity_combi_cols[0][0]+"_English"]+x["Pct_Ethnicity_"+ethinicity_combi_cols[0][1]+"_English"], axis=1)
df_studentNum_pivot_table["Pct_Multi_Ethnicity_Asian+White_Mathematics"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_"+ethinicity_combi_cols[0][0]+"_Mathematics"]+x["Pct_Ethnicity_"+ethinicity_combi_cols[0][1]+"_Mathematics"], axis=1)
df_studentNum_pivot_table["Pct_Avg_Multi_Ethnicity_Asian+White"] = df_studentNum_pivot_table.apply(lambda x: (x["Pct_Ethnicity_"+ethinicity_combi_cols[0][0]+"_English"]+x["Pct_Ethnicity_"+ethinicity_combi_cols[0][0]+"_Mathematics"])/2 + (x["Pct_Ethnicity_"+ethinicity_combi_cols[0][1]+"_English"]+x["Pct_Ethnicity_"+ethinicity_combi_cols[0][1]+"_Mathematics"])/2, axis=1)

df_studentNum_pivot_table["Num_Multi_Ethnicity_Asian+White_English"] = df_studentNum_pivot_table.apply(lambda x: x["Num_Ethnicity_"+ethinicity_combi_cols[0][0]+"_English"]+x["Num_Ethnicity_"+ethinicity_combi_cols[0][1]+"_English"], axis=1)
df_studentNum_pivot_table["Num_Multi_Ethnicity_Asian+White_Mathematics"] = df_studentNum_pivot_table.apply(lambda x: x["Num_Ethnicity_"+ethinicity_combi_cols[0][0]+"_Mathematics"]+x["Num_Ethnicity_"+ethinicity_combi_cols[0][1]+"_Mathematics"], axis=1)
df_studentNum_pivot_table["Num_Avg_Multi_Ethnicity_Asian+White"] = df_studentNum_pivot_table.apply(lambda x: (x["Num_Ethnicity_"+ethinicity_combi_cols[0][0]+"_English"]+x["Num_Ethnicity_"+ethinicity_combi_cols[0][0]+"_Mathematics"])/2 + (x["Num_Ethnicity_"+ethinicity_combi_cols[0][1]+"_English"]+x["Num_Ethnicity_"+ethinicity_combi_cols[0][1]+"_Mathematics"])/2, axis=1)

df_studentNum_pivot_table["Pct_Multi_Ethnicity_Hispanic+Black_English"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_"+ethinicity_combi_cols[1][0]+"_English"]+x["Pct_Ethnicity_"+ethinicity_combi_cols[1][1]+"_English"], axis=1)
df_studentNum_pivot_table["Pct_Multi_Ethnicity_Hispanic+Black_Mathematics"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_"+ethinicity_combi_cols[1][0]+"_Mathematics"]+x["Pct_Ethnicity_"+ethinicity_combi_cols[1][1]+"_Mathematics"], axis=1)
df_studentNum_pivot_table["Pct_Avg_Multi_Ethnicity_Hispanic+Black"] = df_studentNum_pivot_table.apply(lambda x: (x["Pct_Ethnicity_"+ethinicity_combi_cols[1][0]+"_English"]+x["Pct_Ethnicity_"+ethinicity_combi_cols[1][0]+"_Mathematics"])/2 + (x["Pct_Ethnicity_"+ethinicity_combi_cols[1][1]+"_English"]+x["Pct_Ethnicity_"+ethinicity_combi_cols[1][1]+"_Mathematics"])/2, axis=1)

df_studentNum_pivot_table["Num_Multi_Ethnicity_Hispanic+Black_English"] = df_studentNum_pivot_table.apply(lambda x: x["Num_Ethnicity_"+ethinicity_combi_cols[1][0]+"_English"]+x["Num_Ethnicity_"+ethinicity_combi_cols[1][1]+"_English"], axis=1)
df_studentNum_pivot_table["Num_Multi_Ethnicity_Hispanic+Black_Mathematics"] = df_studentNum_pivot_table.apply(lambda x: x["Num_Ethnicity_"+ethinicity_combi_cols[1][0]+"_Mathematics"]+x["Num_Ethnicity_"+ethinicity_combi_cols[1][1]+"_Mathematics"], axis=1)
df_studentNum_pivot_table["Num_Avg_Multi_Ethnicity_Hispanic+Black"] = df_studentNum_pivot_table.apply(lambda x: (x["Num_Ethnicity_"+ethinicity_combi_cols[1][0]+"_English"]+x["Num_Ethnicity_"+ethinicity_combi_cols[1][0]+"_Mathematics"])/2 + (x["Num_Ethnicity_"+ethinicity_combi_cols[1][1]+"_English"]+x["Num_Ethnicity_"+ethinicity_combi_cols[1][1]+"_Mathematics"])/2, axis=1)


#%%
def getColumnNames(cols, category="", pre="", post=""):
    col_list=[]
    for c in cols:
        for p1 in pre:
            for p2 in post:         
                col_list.append(p1+category+c+p2)            
    return col_list     


#%%
# Minors 
ethinicity_minor_cols = ['American Indian or Alaska Native', 'Native Hawaiian or Pacific Islander']

df_studentNum_pivot_table["Pct_Multi_Ethnicity_Minors_English"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_American Indian or Alaska Native_English"]+x["Pct_Ethnicity_Native Hawaiian or Pacific Islander_English"], axis=1)
df_studentNum_pivot_table["Pct_Multi_Ethnicity_Minors_Mathematics"] = df_studentNum_pivot_table.apply(lambda x: x["Pct_Ethnicity_American Indian or Alaska Native_Mathematics"]+x["Pct_Ethnicity_Native Hawaiian or Pacific Islander_Mathematics"], axis=1)
df_studentNum_pivot_table["Pct_Avg_Multi_Ethnicity_Minors"] = df_studentNum_pivot_table.apply(lambda x: (x["Pct_Ethnicity_American Indian or Alaska Native_English"]+x["Pct_Ethnicity_American Indian or Alaska Native_Mathematics"])/2 + (x["Pct_Ethnicity_Native Hawaiian or Pacific Islander_English"]+x["Pct_Ethnicity_Native Hawaiian or Pacific Islander_Mathematics"])/2, axis=1)

df_studentNum_pivot_table["Num_Multi_Ethnicity_Minors_English"] = df_studentNum_pivot_table.apply(lambda x: x["Num_Ethnicity_American Indian or Alaska Native_English"]+x["Num_Ethnicity_Native Hawaiian or Pacific Islander_English"], axis=1)
df_studentNum_pivot_table["Num_Multi_Ethnicity_Minors_Mathematics"] = df_studentNum_pivot_table.apply(lambda x: x["Num_Ethnicity_American Indian or Alaska Native_Mathematics"]+x["Num_Ethnicity_Native Hawaiian or Pacific Islander_Mathematics"], axis=1)
df_studentNum_pivot_table["Num_Avg_Multi_Ethnicity_Minors"] = df_studentNum_pivot_table.apply(lambda x: (x["Num_Ethnicity_American Indian or Alaska Native_English"]+x["Num_Ethnicity_American Indian or Alaska Native_Mathematics"])/2 + (x["Num_Ethnicity_Native Hawaiian or Pacific Islander_English"]+x["Num_Ethnicity_Native Hawaiian or Pacific Islander_Mathematics"])/2, axis=1)

col_deleted = getColumnNames(ethinicity_minor_cols, "Ethnicity_", ["Pct_", "Num_"], ["_English", "_Mathematics"])
df_studentNum_pivot_table.drop(col_deleted, axis=1, inplace=True)

#%% [markdown]
# **2. House Price**
# 
# We extract the significant column, 'House_median', that is expected to be very useful in predicting school scores.

#%%
df_houseprice = pd.pivot_table(df, index=['School Code'], values=['House_median'])
df_houseprice.head()

#%% [markdown]
# **3. Test Score (Target Variable)**

#%%
attr_score = ['Percentage Standard Exceeded', 'Percentage Standard Met', 'Percentage Standard Nearly Met', 'Percentage Standard Not Met']
# attr_remove = ["Subgroup ID", "Student Group", "Category", "County Code", "District Code", "Test Year", "County Name", "District Name", "School Name", "Zip Code"]
# attr_basic = ["School Code", "Test Id"]
df_scores = pd.pivot_table(data=df, index=['School Code', 'Student Groups', 'Category', 'Test Id'], values=attr_score, aggfunc='mean', fill_value=0)
df_scores.reset_index(inplace=True)
df_scores.head()


#%%
df_all_scores = df_scores.loc[(df_scores['Student Groups'] == 'All Students') & (df_scores['Category'] == 'All Students')]
df_scores_pivot_table = df_all_scores.pivot_table(values=attr_score, index='School Code', columns=['Category', 'Student Groups', 'Test Id'])
df_scores_pivot_table.head()


#%%
concat_col_names(df_scores_pivot_table)
df_scores_pivot_table.head()

#%% [markdown]
# We delete substrings "\_All Students" in column names and prepend "Target\_" to score related columns.

#%%
df_scores_pivot_table = df_scores_pivot_table.rename(columns = {col: col.replace("_All Students", "") for col in df_scores_pivot_table.columns})
df_scores_pivot_table.columns = ["Target_"+ col for col in df_scores_pivot_table.columns]

#%% [markdown]
# We can calculate the average scores of English and Mathematics by using `groupby()` function by excluding `Test Id` from the keys and using the `mean()` as an aggregate function.

#%%
df_avg = df_scores.groupby(["School Code", "Category", "Student Groups"], as_index=False).mean().round(2)

#%% [markdown]
# We rename the variables for the readabilitiy, consistency, and understandability.
# As in the Ethinicity, we add the new variable, `Target_Avg_Multi_Percentage Standard Exceeded+Percentage Standard Met`, by combining `Percentage Standard Exceeded` and `Percentage Standard Met`. This variable helps to identify the fairly performance achieving schools.

#%%
df_score_avg = df_avg.copy()
df_score_avg = df_score_avg.loc[(df_score_avg["Category"] == "All Students") & (df_score_avg["Student Groups"] == "All Students")]
df_score_avg=df_score_avg.rename(columns = {'Percentage Standard Exceeded':'Target_Avg_Percentage Standard Exceeded', 
                                  'Percentage Standard Met': 'Target_Avg_Percentage Standard Met', 
                                  'Percentage Standard Nearly Met': 'Target_Avg_Percentage Standard Nearly Met', 
                                  'Percentage Standard Not Met': 'Target_Avg_Percentage Standard Not Met'})
df_score_avg.drop(columns = ["Category", "Student Groups"], axis=1, inplace=True)
#new variable
df_score_avg["Target_Avg_Multi_Percentage Standard Exceeded+Percentage Standard Met"] = df_score_avg['Target_Avg_Percentage Standard Exceeded'] + df_score_avg['Target_Avg_Percentage Standard Met']
df_score_avg.head()

#%% [markdown]
# **4. Merging**
# 
# We combine all dataframe for the school score prediction.

#%%
#merge in an row for each school

df_schools = pd.merge(df_studentNum_pivot_table, df_studentNum_pivot_table_avg, how='left', on="School Code")
                      
df_schools = pd.merge(df_schools, df_houseprice, how='left', on="School Code")

df_schools = pd.merge(df_schools, df_scores_pivot_table, how='left', on="School Code")

df_schools = pd.merge(df_schools, df_score_avg, how='left', on="School Code")


#%%
#df_schools = df_schools.set_index(keys="School Code")
df_schools.shape

#%% [markdown]
# 
# 
# 

#%%
#groupby to make a dataframe (as_index=False)
df_schools_names = df.groupby(["School Code", "School Name", "District Name", "County Name"], as_index=False).mean().round(2)
df_schools_names = df_schools_names.loc[:, ["School Code", "School Name", "District Name", "County Name"]]
df_schools_names.head()


#%%
#df_schools.reset_index(drop=True).head()
df_schools.head()


#%%

#merge hen need to see the school names
df_schools_names_all = pd.merge(df_schools_names, df_schools, how='left', on="School Code")
df_schools_names_all.head()


#%%
# school_name_columns = ["School Name", "District Name", "County Name"]

# dataframe selection with multiple columns
# df_schools_names_all[school_name_columns + ["Avg_Percentage Standard Exceeded"]]
# df_schools[["Num_Ethnicity_Black or African American_Mathematics", "House_median", "Percentage Standard Exceeded_Mathematics"]]


#%%
#max value
#df_schools[["Num_Ethnicity_Black or African American_Mathematics"]].max()

#df_schools.loc[df_schools["Num_Ethnicity_Black or African American_Mathematics"].idxmax(), "Percentage Standard Exceeded_Mathematics"]
#df_schools.loc[[df_schools["Num_Ethnicity_Black or African American_Mathematics"].idxmax()]]

#List of data of the best performance students for each student groups
# idx = final_data_school.groupby(["Category", "Student Groups"])["Percentage Standard Exceeded"].transform(max) == final_data_school["Percentage Standard Exceeded"]
# final_data_school[idx].head()


#%%
#free space
del df_studentNum_pivot_table
del df_studentNum_pivot_table_avg
del df_houseprice
del df_scores_pivot_table
del df_score_avg
del df_scores

#%% [markdown]
# ### Imputing Missing Values 
# 
# Before we put features into a model, missing values must be filled and all features must be encoded.  
# 
# For various reasons, many real world datasets contain missing values, often encoded as blanks, NaNs or other placeholders. Such datasets however are incompatible with scikit-learn estimators which assume that all values in an array are numerical, and that all have and hold meaning. A basic strategy to use incomplete datasets is to discard entire rows and/or columns containing missing values. However, this comes at the price of losing data which may be valuable (even though incomplete). A better strategy is to impute the missing values, i.e., to infer them from the known part of the data.   
# https://scikit-learn.org/stable/modules/impute.html  
# 
# To deal with the missing values, we use the basic strategies for imputing missing values. Missing values are imputed using the statistics of the *mean* of each column in which the missing values are located. 

#%%
from sklearn.impute import SimpleImputer
#     imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#     imputer = imputer.fit(X[:, 1:3])
#     X[:, 1:3] = imputer.transform(X[:, 1:3])

def replace_missing_value(df, number_features):

    imputer = SimpleImputer(strategy = 'mean')
    df_num = df[number_features]
    imputer.fit(df_num)
    X = imputer.transform(df_num)
    res_def = pd.DataFrame(X, columns=df_num.columns)
    return res_def


#%%
df_schools = replace_missing_value(df_schools, df_schools.columns)
#df_schools.loc[df_schools["School Name"] == "Mission Education Center"]
df_schools

#%% [markdown]
# Reference   
# 15 Statistical Hypothesis Tests in Python: 
# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
# 
# ## Statistical Hypothesis Testing
# 
# ### T-Test for means of two independent samples
# 
# Now we perform a hypothesis test (two-sample test) by assuming two group of samples are independent.  
# 
# - $H_0$: There is **no difference** in students' scores between sample1 and sample2.
# - $H_1$: There **exist difference** in students' scores between sample1 and sample2.
# - $\alpha$ = 0.05

#%%
def ttest_ind_two(category1, group1, category2, group2, target_col):
    df_sample1 = df.loc[(df['Category'] == category1) & (df['Student Groups'] == group1)][target_col]
    df_sample2 = df.loc[(df['Category'] == category2) & (df['Student Groups'] == group2)][target_col]
    t, p_val = stats.ttest_ind(df_sample1, df_sample2)
    return t, p_val

#%% [markdown]
# Here is the student group information `studentGroup_types` we try to investigate. 

#%%
#get dataframe using groupby
# len(df.groupby(["Category", "Student Groups"]).groups) #47
# print("Number of Category: {}".format(df["Category"].nunique())) #10
# print("Number of Category: {}".format(df["Category"].value_counts())) 

studentGroup_types = df.groupby(["Category", "Student Groups"], as_index=False).mean()
studentGroup_types = studentGroup_types.loc[:, ["Category", "Student Groups"]]
studentGroup_types #(47, 2)

#%% [markdown]
# 'Category' = ['Student Groups']
# * "All Students" = [All Students]
# * "Gender" = [Male, Female]  
# * "Ethnicity" = [American Indian or Alaska Native, Asian, Black or African American, Filipino, Hispanic or Latino, Native Hawaiian or Pacific Islander, Two or more races, White]
# * "English-Language Fluency" = [English learner, English learners (ELs) enrolled in school in the U.S. fewer than 12 months, English learners enrolled in school in the U.S. 12 months or more, English only, Ever-ELs, Fluent English proficient and English only, Initial fluent English proficient (IFEP), Reclassified fluent English proficient (RFEP), To be determined (TBD)]
# * "Parent Education" = [College graduate, Declined to state, Graduate school/Post graduate, High school graduate, Not a high school graduate, Some college (includes AA degree)]
# * "Migrant" = [Migrant education]
# * "Economic Status" = [Economically disadvantaged, Not economically disadvantaged]
# * "Disability Status" = [Students with disability, Students with no reported disability]
# * "Ethnicity for Economically Disadvantaged" = [American Indian or Alaska Native, Asian, Black or African American, Filipino, Hispanic or Latino, Native Hawaiian or Pacific Islander, Two or more races, White]
# * Ethnicity for Not Economically Disadvantaged = [American Indian or Alaska Native, Asian, Black or African American, Filipino, Hispanic or Latino, Native Hawaiian or Pacific Islander, Two or more races, White]
#%% [markdown]
# We tests whether the means of two independent samples are significantly different.
# If there is no difference (p-value is greater or equal than $\alpha$ 0.05), then we want to **eliminate or merge** that student group information due to too much generating features.
#%% [markdown]
# **T-Test for all pairs of student group information**

#%%
print("T-test for the means of two samples for on target value '{}''".format(target_col))
l = len(studentGroup_types)
no_diff = [] #list of ('Category', 'Student Group') that have p-value of greater or equal to 0.05
for i in range(0,l):
    for j in range(i+1,l):
        category1 = studentGroup_types.iloc[i]["Category"]
        group1 = studentGroup_types.iloc[i]["Student Groups"]
        category2 = studentGroup_types.iloc[j]["Category"]
        group2 = studentGroup_types.iloc[j]["Student Groups"]

        t, p_val = ttest_ind_two(category1, group1, category2, group2, target_col)
        if p_val > 0.05: #no difference (cannot reject H0) 
            #print((category1, group1, category2, group2+": "))
            #print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))
            no_diff.append((category1, group1))
            no_diff.append((category2, group2))

#%% [markdown]
# We sorted the student group information on the order of the occurrences. The following shows the results with the counter number. We can use this results when getting rid of the student group information for generating less number of features. In other words, we want to generate and use the features that strongly affect for predicting the school scores.

#%%
# for c in Counter(no_diff).most_common(1):
#     print(c[0][0], c[0][1])

#Counter(no_diff) 
#Counter(no_diff).most_common(1)[0] #(('Ethnicity for Not Economically Disadvantaged','Native Hawaiian or Pacific Islander'), 14)
#Counter(no_diff).most_common(1)[0][0] #(Category, Student Groups)
#Counter(no_diff).most_common(1)[0][1] #Counter number, ex) 14
#Counter(no_diff).most_common(1)[0][0][0] #Category
#Counter(no_diff).most_common(1)[0][0][1] #Student Groups

Counter(no_diff).most_common()

#%% [markdown]
# **T-Test between subjects (English and Mathematics)**

#%%
#no_diff_subject = []
print("Among {} groups, the below listed groups do not show the score differences between subjects of English and Mathematics.".format(studentGroup_types.shape[0]))

def ttest_ind_two_subject(category, group, target_col):
    df_sample1 = df.loc[(df['Category'] == category) & (df['Student Groups'] == group) & (df['Test Id'] == 'English')][target_col]
    df_sample2 = df.loc[(df['Category'] == category) & (df['Student Groups'] == group) & (df['Test Id'] == 'Mathematics')][target_col]
    t, p_val = stats.ttest_ind(df_sample1, df_sample2)
    if(p_val > 0.05):
        print("({} : {}), t-test: {:.10f}, p-value: {:.10f}".format(category, group, t, p_val))
        #no_diff_subject.append((category, group))

for index, row in studentGroup_types.iterrows():
    ttest_ind_two_subject(row['Category'], row['Student Groups'], target_col)

#%% [markdown]
# ==> The score differences exist in most of the groups. However, we may eliminate the subjects (`Test Id`) for further analysis or constructing prediction models, because the subject difference is not our major concerns.
#%% [markdown]
# #### **Specific interest pairs of student groups**
# 
# We again analyzed the two samples (where we have special interests in those relations) using the T-test.

#%%
#Gender
print('Gender')
t, p_val = ttest_ind_two('Gender', 'Male', 'Gender', 'Female', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

#Ethinity
print('Ethnicity')
t, p_val = ttest_ind_two('Ethnicity', 'Asian', 'Ethnicity', 'White', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

t, p_val = ttest_ind_two('Ethnicity', 'Asian', 'Ethnicity', 'Filipino', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

t, p_val = ttest_ind_two('Ethnicity for Economically Disadvantaged', 'Asian', 'Ethnicity', 'Asian', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

t, p_val = ttest_ind_two('Ethnicity for Economically Disadvantaged', 'Asian', 'Ethnicity', 'Asian', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

print('Minor Ethnicity')
t, p_val = ttest_ind_two('Ethnicity', 'Black or African American', 'Ethnicity', 'Native Hawaiian or Pacific Islander', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

t, p_val = ttest_ind_two('Ethnicity', 'American Indian or Alaska Native', 'Ethnicity', 'Native Hawaiian or Pacific Islander', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

t, p_val = ttest_ind_two('Ethnicity', 'Black or African American', 'Ethnicity', 'American Indian or Alaska Native', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))


#English-Language Fluency
print('English-Language Fluency')
t, p_val = ttest_ind_two('English-Language Fluency', 'Initial fluent English proficient (IFEP)', 'English-Language Fluency', 'Reclassified fluent English proficient (RFEP)', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

t, p_val = ttest_ind_two('English-Language Fluency', 'Initial fluent English proficient (IFEP)', 'English-Language Fluency', 'English only', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

t, p_val = ttest_ind_two('English-Language Fluency', 'Reclassified fluent English proficient (RFEP)', 'English-Language Fluency', 'English only', target_col)
print("t: {:.10f}, p-value: {:.10f}".format(t, p_val))

#%% [markdown]
# ==> In these test, when the  $p-$value is much smaller than  𝛼  = 0.05, and we reject the null hypothesis that there is no difference. In fact, the $p-$values zero indicates that there is significant differences between two samples in scores. 
#%% [markdown]
# **Stepwise way of deletion**
# 
# We can choose to drop all the rows of ('Category', 'Student Group') that do not actually affect a target variable, 'Percentage Standard Exceeded' or 'Percentage Standard Not Met' using the **stepwise way** for dropping the student group information.

#%%
def stepwise_deletion_ttest_ind(category, student_group, target_col, studentGroup_type):
    while True:
        l = len(studentGroup_types)
        no_diff = [] #list of ('Category', 'Student Group') that have p-value of greater or equal to 0.05
        for i in range(0,l):
            for j in range(i+1,l):
                category1 = studentGroup_types.iloc[i][category]
                group1 = studentGroup_types.iloc[i][student_group]
                category2 = studentGroup_types.iloc[j][category]
                group2 = studentGroup_types.iloc[j][student_group]

                t, p_val = ttest_ind_two(category1, group1, category2, group2, target_col)
                if p_val > 0.05: #no difference            
                    no_diff.append((category1, group1))
                    no_diff.append((category2, group2))
        if no_diff == []:
            break

        not_relevent_feat = Counter(no_diff).most_common(1)
        print(not_relevent_feat)
        #not_relevent_feat[0][0][0]: Category, not_relevent_feat[0][0][1]: Student Groups
        idx = studentGroup_types.index[(studentGroup_types[category] == not_relevent_feat[0][0][0]) & (studentGroup_types[student_group] == not_relevent_feat[0][0][1])]
        studentGroup_types.drop(idx, axis=0, inplace=True)

#%% [markdown]
# ==> We analyzed all pairs of two samples using T-test and found the two samples that have no difference (p-value is greater or equal than $\alpha$=0.05). Then, we select the most occurrence of student group information and delete that row from investigating student group information table (`studentGroup_types`). 
# 
# We then reiterate the T-test process for find and delete next least affecting factor. The following results shows the deleted student group information for every step with the number of occurrences.
# 
# [(('Ethnicity for Not Economically Disadvantaged', 'Native Hawaiian or Pacific Islander'), 14)]  
# [(('English-Language Fluency', 'To be determined (TBD)'), 10)]  
# [(('Ethnicity for Not Economically Disadvantaged', 'American Indian or Alaska Native'), 9)]  
# [(('Ethnicity for Economically Disadvantaged', 'Native Hawaiian or Pacific Islander'), 4)]  
# [(('English-Language Fluency', 'English learners (ELs) enrolled in school in the U.S. fewer than 12 months'), 4)]  
# [(('Ethnicity for Economically Disadvantaged', 'American Indian or Alaska Native'), 3)]  
# [(('English-Language Fluency', 'English only'), 3)]  
# [(('Ethnicity for Not Economically Disadvantaged', 'Hispanic or Latino'), 2)]  
# [(('Ethnicity', 'White'), 2)]  
# [(('Ethnicity', 'American Indian or Alaska Native'), 2)]  
# [(('Ethnicity', 'Black or African American'), 2)]  
# [(('English-Language Fluency', 'English learner'), 1)]  
# [(('English-Language Fluency', 'Reclassified fluent English proficient (RFEP)'), 1)]  
# [(('Ethnicity', 'Filipino'), 1)]  
# [(('Ethnicity', 'Native Hawaiian or Pacific Islander'), 1)]  
# [(('Ethnicity for Economically Disadvantaged', 'Black or African American'), 1)]  
# [(('Ethnicity for Economically Disadvantaged', 'Two or more races'), 1)]  
# [(('Ethnicity for Not Economically Disadvantaged', 'Black or African American'), 1)]
#%% [markdown]
# ## Variables for Modeling
# 
# ### Decision for Variables
#%% [markdown]
# ==> Based on the T-test, we can eliminate or merge these the weak affecting student group indicators.
# By referring top indicators in `no_diff`, we adjust the following indicators for making a machine-learning based school score prediction model.
# 
# 1) Delete the meaningless indicators such as, 'To be determined (TBD)' and 'Declined to state'.  
# 2) Delete the 'Disability Status', 'Economic Status', 'Ethnicity for Economically Disadvantaged', 'Ethnicity for Not Economically Disadvantaged'. It seems redundant and rather trivial that do not produce the new results.  
# 3) For 'Ethnicity', delete 'Two or more races', merge "Native Hawaiian or Pacific Islander" and "American Indian or Alaska Native", and create `Minor` races. 
# - `Pct_Avg_Multi_Ethnicity_Minor`, `Pct_Multi_Ethnicity_Minor_English`, and `Pct_Multi_Ethnicity_Minor_Mathematics`    
# 
# 4) For 'English-Language Fluency', we organize the indicators as follows.    
# - Delete 'English learners (ELs) enrolled in school in the U.S. fewer than 12 months' and 'English learners enrolled in school in the U.S. 12 months or more' and use the 'English learner' only instead  
# - Delete 'Ever-ELs' which indicates 'Reclassified fluent English proficient (RFEP)' + 'English learner' 
# 
# 5) For 'Parent Education', delete 'Some college (includes AA degree)'  

#%%
def getColumns(df, name, logic="AND"):
    col_list = []  
    for col in df.columns:
        if(logic == "AND"):
            if all(s in col for s in name): 
                col_list.append(col)
        elif(logic == "OR"):
            if any(s in col for s in name):
                col_list.append(col)
            
    return col_list


#%%
# df.drop(df[df["Student Groups"].str.contains("Declined", regex=False, case=False, na=False)].index, axis=0, inplace=True)
# idx = df.loc[(df['Category'] == 'English-Language Fluency') & (df['Student Groups'] == 'To be determined (TBD)')].index
# idx = df.index[df['Category'] == 'Ethnicity for Economically Disadvantaged']
# df.drop(idx, axis=0, inplace=True)

#1) Delete the meaningless indicators such as, 'To be determined (TBD)' and 'Declined to state'.
df_schools.drop(getColumns(df_schools, ["TBD"], "OR"), axis=1, inplace=True)
df_schools.drop(getColumns(df_schools, ["Declined"], "OR"), axis=1, inplace=True)

#2) Delete the 'Disability Status', 'Economic Status', 'Ethnicity for Economically Disadvantaged', 'Ethnicity for Not Economically Disadvantaged'.
#todo: delete or not
#df_schools.drop(getColumns(df_schools, ["Disability Status"], "OR"), axis=1, inplace=True)
#df_schools.drop(getColumns(df_schools, ["Economic Status"], "OR"), axis=1, inplace=True)
df_schools.drop(getColumns(df_schools, ["Ethnicity for Economically Disadvantaged"], "OR"), axis=1, inplace=True)
df_schools.drop(getColumns(df_schools, ["Ethnicity for Not Economically Disadvantaged"], "OR"), axis=1, inplace=True)

#3) For 'Ethnicity', delete 'Two or more races'
df_schools.drop(getColumns(df_schools, ["Two or more race"], "OR"), axis=1, inplace=True)
df_schools["Pct_Avg_Multi_Ethnicity_Minor"] = df_schools.apply(lambda x: x["Pct_Avg_Ethnicity_Native Hawaiian or Pacific Islander"]+x["Pct_Avg_Ethnicity_Native Hawaiian or Pacific Islander"], axis=1)
df_schools.drop(getColumns(df_schools, ["Native Hawaiian or Pacific Islander", "American Indian or Alaska Native"], "OR"), axis=1, inplace=True)

#4) For 'English-Language Fluency'
df_schools.drop(getColumns(df_schools, ["enrolled in school in the U.S."], "OR"), axis=1, inplace=True)
df_schools.drop(getColumns(df_schools, ["Ever-ELs"], "OR"), axis=1, inplace=True)


#5) For 'Parent Education
df_schools.drop(getColumns(df_schools, ["Some college"], "OR"), axis=1, inplace=True)


#%%
def getPerformedSchools(df, score_attr, relate, percent):
    #df_level = df_schools.loc[df_schools["Target_Avg_Percentage Standard Not Met"] > 80]
    df_level = df.loc[ops[relate](df[score_attr], percent)]
    level_index = df_level.index
    #df_level = pd.merge(df_schools_names, df_level, how='right', on="School Code")
    df_level = pd.merge(df_schools_names, df_level, how='right', on="School Code")
    print("The number of schools that ['{}' {} {}] is {}.".format(score_attr, relate, percent, df_level.shape[0]))
    return df_level, level_index 

#%% [markdown]
# ### Independent and Dependent Variables
# 
# **For predictors (Independent variables):**    
# 
# Organized variables: 
# - ['Num'] x ['Category' + 'Student Groups' + 'Test Id']: 
#     - 'Num': 'Students with Scores' (Number of students)
#     - 'Test Id' = [English, Mathematics]
#     - 'Category' and 'Student Groups': See `studentGroup_types`
# - House_median: House median prices in the school zones
#     
# New variables:
# - ['Pct'] x ['Category' + 'Student Groups' + 'Test Id']: 
#     - 'Pct': Percentage of students over all students in a school
# - (['Num'] | ['Pct']) x ['Avg' + 'Category' + 'Student Groups']:
#     - 'Avg' : Average number of percentage of students for English and Mathematics
#         - (English + Mathematics) / 2
# - ['Pct'] x (['Multi' + 'Test Id'] | ['Avg' + 'Multi']):
#     - 'Multi':
#         - 'Asian+White' or 'Hispanic+Black' in 'Ethnicity'
#         - 'Minor' : American Indian or Alaska Native' and 'Native Hawaiian or Pacific Islander' in 'Ethnicity'
# 
# **Target variable (Dependent variable):**
# 
# Continuous:
# - Average percentage (Target_Avg) for all four achievement levels:
#     - 'Percentage Standard Exceeded': Exceeded (Level 4)
#     - 'Percentage Standard Met': Standard (Level 3)
#     - 'Percentage Standard Nearly Met': Nearly (Level 2)
#     - 'Percentage Standard Not Met': NotMet (Level 1)  
# - ['Target_Avg_Multi_Percentage Standard Exceeded+Percentage Standard Met']:
#     - Sum of two levels (Level 4 + Level 3) that can represent the portions that achieve the standards in a school.
# 
# Ordinal:
# - Ranking (1st, 2nd, last --> Ranked variable) 
# 
# Categorical:
# - Need Help/No Need Help label (Classification)

#%%
# Ordinal 
df_schools["Rank_Level4"] = df_schools["Target_Avg_Percentage Standard Exceeded"].rank(ascending=False) # creating a rank column and passing the returned rank series 
df_schools["Rank_Level1"] = df_schools["Target_Avg_Percentage Standard Not Met"].rank(ascending=False) # creating a rank column and passing the returned rank series 

#df_schools.sort_values("Target_Avg_Percentage Standard Exceeded", inplace = True, ascending=False)

# Categorical : Need Help/No Need Help label (Classification)
df_schools['NeedHelp'] = 0
df_needHelp, df_needHelpidx = getPerformedSchools(df_schools, "Target_Avg_Percentage Standard Not Met", '>', 80)
df_schools.loc[df_needHelpidx, 'NeedHelp'] = 1


#%%
df_schools = df_schools.sort_index()

#%% [markdown]
# **Final Independent Variables and Target Variable**

#%%
#independent columns
#delete: Keywords
X = df_schools.drop(getColumns(df_schools, ["English", "Mathematics"], "OR"), axis=1)
X_Num = df_schools.drop(getColumns(df_schools, ["Pct", "English", "Mathematics"], "OR"), axis=1)
X_Pct = df_schools.drop(getColumns(df_schools, ["Num", "English", "Mathematics"], "OR"), axis=1)

#delete: specific column name
X = X.drop(['School Code'], axis=1)
X_Num = X_Num.drop(['School Code'], axis=1)
X_Pct = X_Pct.drop(['School Code'], axis=1)

X_all = X.copy()
X_all_Num = X_Num.copy()
X_all_Pct = X_Pct.copy()

# #Include
# X = X[getColumns(X, ["Avg"])]

#target column 
#y = df_schools.iloc[:,-1]
#round(0) for chi square analysis
target_var_name = "Target_Avg_"+target_col
X = X.drop([target_var_name], axis=1)
y = df_schools[target_var_name].round(0)

#y = y.astype('float64') # Your y is of type object, so sklearn cannot recognize its type. 
print("Independent variables (size:{})".format(X.columns.size))
print(X.columns)
print("Target variable: ", target_var_name)

#%% [markdown]
# ## Correlation Tests
# 
# ### Matrix with Heatmap

#%%
from matplotlib.colors import LinearSegmentedColormap

def corrLinearSegmentedColormap(df, sortedKey):
    corr_sorted = abs(df[sortedKey]).sort_values()
    sorted_df = df_schools[list(corr_sorted.index)]
    corr = round(sorted_df.corr(), 2)

    min_color = 'white'
    max_color = (0.03137254, 0.18823529411, 0.41960784313, 1)
    cmap = LinearSegmentedColormap.from_list("", [max_color,
                                                  min_color,
                                                  max_color])
    fig = sns.heatmap(corr, annot=True, cmap=cmap,
                      xticklabels=corr.columns.values,
                      yticklabels=corr.columns.values,
                      cbar=False)
    plt.xticks(rotation=90)
    fig.xaxis.set_tick_params(labelsize=10)
    fig.yaxis.set_tick_params(labelsize=10)

    plt.show()

#%% [markdown]
# Correlation states how the features are related to each other or the target variable.
# 
# Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable).
# 
# Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.
#%% [markdown]
# **Correlation Table with Number Related Features**

#%%
# top_corr_features = corr.index
corr = X_all_Num.corr()
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df_schools[corr.index].corr(), annot=True, cmap="RdYlGn")

#%% [markdown]
# **Correlation Table with Percent Related Features**

#%%
corr = X_all_Pct.corr()
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df_schools[corr.index].corr(), annot=True, cmap="RdYlGn")

#%% [markdown]
# ==> With the results above, just as assumed, the high score (`Target_Avg_Percentage Standard Exceeded`) is correlated to higher the house price (`House_median`), the higher education (`Num_Avg_Parent Education_Graduate school/Post graduate`), and good economic status (`Num_Avg_Economic Status_Not economically disadvantaged`).  
# 
# It is interesting that the number of Hispanics (`Num_Avg_Ethinicity_Hispanic or Latino`) is highly correlated (**0.94**) with the number of economically disadvantaged students. The percent of Hispanics (`Pct_Avg_Ethinicity_Hispanic or Latino`) is correlated (**0.78**) but not as strong as the number feature.
# In California, there is the largest number of Hispanic students among other ethnicity students, and this can be the cause of the high correlation.

#%%
#Number of students in each ethnicity
df_ethnicity = df.groupby(["Category", "Student Groups"], as_index=False).count().round(2)


#%%
df_ethnicity = df_ethnicity.loc[(df_ethnicity["Category"] == "Ethnicity"), ["Category", "Student Groups", "School Name"]]
df_ethnicity

#%% [markdown]
# ### Pearson’s Correlation Coefficient
# 
# Pearson's correlation coefficient tests whether two samples have a linear relationship.
# 
# Assumptions:
# - Observations in each sample are independent and identically distributed.
# - Observations in each sample are normally distributed.
# - Observations in each sample have the same variance.
# 
# Interpretation:
# - $H_0$: The two samples are independent.
# - $H_1$: There is a dependency between the samples.
#%% [markdown]
# Here we calculate Pearsons’s Correlation Using SciPy, `scipy.stats.pearsonr(x, y)`.  
# For example, we investigated the relationship between 'Pct_Ethnicity_Asian_Mathematics' and 'Target_Percentage Standard Exceeded_Mathematics'.

#%%
corr, p_val = stats.pearsonr(df_schools['Pct_Ethnicity_Asian_Mathematics'], df_schools['Target_Percentage Standard Exceeded_Mathematics'])
print("* Spearman Rank Correlation between '{}' and '{}':\n corr: {:.10f}, p-value: {:.10f}".format('Pct_Ethnicity_Asian_Mathematics', 'Target_Percentage Standard Exceeded_Mathematics', corr, p_val))

#%% [markdown]
# ==> We reject the null hypothesis $H_0$.  
# The portion of Asian students and the higher scores in Mathematics is not independent but strongly correlated.
#%% [markdown]
# ### Spearman's Rank Correlation
# 
# Spearman's correlation measures the strength and direction of monotonic association between two variables. Spearman’s rank correlation is the Pearson’s correlation coefficient of the ranked version of the variables. We can define a function for calculating the spearman's rank correlation. 
# 
# Assumptions:
# Observations in each sample are independent and identically distributed.
# Observations in each sample can be ranked.
# 
# Interpretation:
# - $H_0$: The two samples are independent.
# - $H_1$: There is a dependency between the samples.

#%%
# Create a function that takes in x's and y's
def spearmans_rank_corr(xs, ys):
    
    # Calculate the rank of x's
    xranks = pd.Series(xs).rank()
    
    # Caclulate the ranking of the y's
    yranks = pd.Series(ys).rank()
    
    # Calculate Pearson's correlation coefficient on the ranked versions of the data
    return scipy.stats.pearsonr(xranks, yranks)

#%% [markdown]
# Here we calculate Spearman’s Rank Correlation Using SciPy, `scipy.stats.spearmanr(x, y)`.  
# For example, we investigated the relationship between 'House_median' and 'Target_Avg_Percentage Standard Exceeded'.

#%%
corr, p_val = stats.spearmanr(df_schools['House_median'], df_schools['Target_Avg_Percentage Standard Exceeded'])
print("* Spearman Rank Correlation between '{}' and '{}':\n corr: {:.10f}, p-value: {:.10f}".format('House_median', 'Target_Avg_Percentage Standard Exceeded', corr, p_val))

#%% [markdown]
# ====> We reject the null hypothesis $H_0$.  
# The house prices and high scores is not independent but correlated.
#%% [markdown]
# ## Feature Selection
# 
# Reference  
# 
# Feature Selection Techniques in Machine Learning with Python  
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
# 
# Feature selection using statistical testing:
# https://medium.com/@vadim_uvarov/feature-selection-using-statistical-testing-2d8e7b5e27b8
#%% [markdown]
# ### Univariate Selection  
# 
# Statistical tests can be used to select those features that have the strongest relationship with the output variable.
# 
# The scikit-learn library provides the [SelectKBest class](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest)  that can be used with a suite of different statistical tests to select a specific number of features.
# 
# We use the chi-squared (chi$^2$) statistical test for non-negative features to select 20 best features.

#%%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'Score']  #naming the dataframe columns
print("{} largest scores and features:".format(20))
print(featureScores.nlargest(20, 'Score'))  #print 20 best features
KBestFeatures = featureScores.nlargest(20, 'Score')
KBestFeatures['Feature'].tolist()

#%% [markdown]
# ==> As expected, for the higher achievement (`Percentage of Standard Exceeded`),  higher house prices, higher economic status, Asians and Whites in Ethnicity, and higher education.
# `Rank_Level4` is derived from the `Percentage of Standard Exceeded`, so it must be strongly correlated.
#%% [markdown]
# ### Feature Importance
# 
# We obtain the feature importance of each feature of the dataset by using the feature importance property of the model.
# 
# Feature importance gives you a score for each feature of the data, the higher the score more important or relevant is the feature towards the output variable.
# 
# Feature importance is an inbuilt class that comes with Tree Based Classifiers, we will be using Extra Tree Classifier for extracting the top 20 features for the dataset.

#%%
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(n_estimators=100)
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#%% [markdown]
# ==> The `Rank_Level4` and the other score variables are identified as the important features.
# 
# ==> Finally, we identify the important features as below. These features will be used for building machine learning models later.

#%%
#Concatenation and remove duplicates
list1 = KBestFeatures['Feature'].tolist()
list2 = feat_importances.index.tolist()

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

#features selection results : intersection
attr_important_features_inter = intersection(list1, list2)
attr_important_features_inter = list(set(attr_important_features_inter + attr_avg_score))
#features selection results : union
attr_important_features = list(set(list1+list2)) #len(attr_important_features): 31

#%% [markdown]
# ## Further Analysis for Recommendations
# 
# ### Analysis on Specific Groups

#%%
df_schools.shape


#%%
school_names = ["School Name", "District Name", "County Name"]
interest_cols1 = getColumns(df_schools, ["House", "Rank"], "OR")
interest_cols2 = getColumns(df_schools, ["Pct", "Ethnicity", "Avg"], "AND")
interest_cols3 = getColumns(df_schools, ["Pct", "Education", "Avg"], "AND")
interest_cols4 = getColumns(df_schools, ["Target", "Avg"], "AND")
#remove item contains specific word 
#interest_cols2 = [x for x in interest_cols2 if "Multi" not in x] 
#https://stackoverflow.com/questions/18790072/how-to-remove-items-from-a-list-that-contains-words-found-in-items-in-another-li
#remove item contains specific word list #[sa for sa in a if not any(sb in sa for sb in b)]
del_cols = ["Filipino"]
interest_cols2 = [x for x in interest_cols2 if not any(y in x for y in del_cols)]

interest_cols_final = school_names + interest_cols1 + interest_cols2 + interest_cols3 + interest_cols4 


#%%
recommendation_model_feat = ['Pct_Avg_Multi_Ethnicity_Asian+White',
                             'Pct_Avg_Multi_Ethnicity_Hispanic+Black', 'House_median', 
                             'Pct_Avg_Parent Education_Graduate school/Post graduate', 
                             'Pct_Avg_Parent Education_Not a high school graduate',
                             'Pct_Avg_Parent Education_No_College',
                             'Pct_Avg_Economic Status_Economically disadvantaged']

#%% [markdown]
# * Top and bottom 5%  
# 8768 * 0.05 $\approx$ = 450
# 
# The number of schools that ['Target_Avg_Percentage Standard Not Met' > 90] is 96.

#%%
df_Level1, idx_Level1 = getPerformedSchools(df_schools, "Target_Avg_Percentage Standard Not Met", '>', 80)
#df_Level1 = pd.merge(df_schools_names, df_Level1, how='right', on="School Code")
df_Level1['Pct_Avg_Parent Education_No_College'] = df_Level1['Pct_Avg_Parent Education_High school graduate'] + df_Level1['Pct_Avg_Parent Education_Not a high school graduate'] 

#%% [markdown]
# ==> The achivement performance inferior groups is much large and extreme.   
# Many of those schools have zero percent of 'Percentage of Standard Exceeded' students.

#%%
df_Level1_rank = df_schools.loc[df_schools["Rank_Level1"] < 450]
df_Level1_rank = pd.merge(df_schools_names, df_Level1_rank, how='right', on="School Code")
df_Level1_rank['Pct_Avg_Parent Education_No_College'] = df_Level1_rank['Pct_Avg_Parent Education_High school graduate'] + df_Level1_rank['Pct_Avg_Parent Education_Not a high school graduate'] 


#%%
#df_Level1_rank.loc[df_Level1_rank["House_median"] > 1500000]


#%%
#df_Level1.loc[df_Level1["House_median"] < 200000][interest_cols_final].head()


#%%
df_Level1[school_names + recommendation_model_feat]

#%% [markdown]
# df_Level4, idx_Level4 = getPerformedSchools(df_schools, "Target_Avg_Percentage Standard Exceeded", '>', 80)
# The number of schools that ['Target_Avg_Percentage Standard Exceeded' > 80] is 7.
# 
# ==> Only seven schools have more than 80\% of Percentage Standard Met students in the average of both English and Mathematics.

#%%
df_Level4, idx_Level4 = getPerformedSchools(df_schools, "Target_Avg_Percentage Standard Exceeded", '>', 59)
df_Level4['Pct_Avg_Parent Education_No_College'] = df_Level4['Pct_Avg_Parent Education_High school graduate'] + df_Level4['Pct_Avg_Parent Education_Not a high school graduate'] 


#%%
df_Level4_rank = df_schools.loc[df_schools["Rank_Level4"] < 450]
df_Level4_rank = pd.merge(df_schools_names, df_Level4_rank, how='right', on="School Code")
df_Level4_rank['Pct_Avg_Parent Education_No_College'] = df_Level4_rank['Pct_Avg_Parent Education_High school graduate'] + df_Level4_rank['Pct_Avg_Parent Education_Not a high school graduate'] 


#%%
df_Level4[school_names + recommendation_model_feat]


#%%
def getNonzeroDF(df, feature):
    df_feat = df.loc[df[feature]!=0][feature]
    return df_feat


#%%
#feature: list
def getNonzeroDF2(df, feature):
#    return [index for row, index in df.iterrows() for f in feature if all(row[f]==0)]
    for row, index in df.iterrows():
        if all(df[row][f]==0 for f in feature):
            print(index)


#%%
df_Level1_no0 = df_Level1[recommendation_model_feat]
df_Level4_no0 = df_Level4[recommendation_model_feat]
df_Level1_no0


#%%
#delete meaningless data
education_cond = ((df_Level1_no0['Pct_Avg_Parent Education_Graduate school/Post graduate']==0)&
                  (df_Level1_no0['Pct_Avg_Parent Education_Not a high school graduate']==0)&
                  (df_Level1_no0['Pct_Avg_Parent Education_No_College']==0)) 
economic_cond = (df_Level1_no0['Pct_Avg_Economic Status_Economically disadvantaged']==0)

ethinicity_cond = ((df_Level1_no0['Pct_Avg_Multi_Ethnicity_Asian+White']==0)&
                   (df_Level1_no0['Pct_Avg_Multi_Ethnicity_Hispanic+Black']==0)) 


education_cond_4 = ((df_Level4_no0['Pct_Avg_Parent Education_Graduate school/Post graduate']==0)&
                        (df_Level4_no0['Pct_Avg_Parent Education_Not a high school graduate']==0)&
                          (df_Level4_no0['Pct_Avg_Parent Education_No_College']==0)) 


del_idx = df_Level1_no0[ethinicity_cond | education_cond | economic_cond].index
del_idx4 = df_Level4_no0[education_cond_4].index


#%%
df_Level1_no0.drop(del_idx, axis=0, inplace=True)
df_Level4_no0.drop(del_idx4, axis=0, inplace=True)


#%%
df_Level1_no0 = df_Level1_no0.reset_index(drop=True)
df_Level4_no0 = df_Level4_no0.reset_index(drop=True)
# recommend_df = df_Level1_no0.copy()
# recommend_df['urgentHelp'] = 1
#recommend_df


#%%
#'UrgentHelp': 0 (No), 1 (Yes)
df_Level4_no0['UrgentHelp'] = '0'
df_Level1_no0['UrgentHelp'] = '1'
df_recommendation = pd.concat([df_Level4_no0, df_Level1_no0], axis=0)

#%% [markdown]
# **Decision Tree**

#%%
y = df_recommendation['UrgentHelp']

recommendation_model_feat2 = [#'Pct_Avg_Multi_Ethnicity_Asian+White',
                             #'Pct_Avg_Multi_Ethnicity_Hispanic+Black', 
                             # 'House_median', 
                             # 'Pct_Avg_Parent Education_Graduate school/Post graduate', 
                             #'Pct_Avg_Parent Education_Not a high school graduate',
                             'Pct_Avg_Parent Education_No_College',
                             #'Pct_Avg_Economic Status_Economically disadvantaged'
                             ]

X = df_recommendation.loc[:, recommendation_model_feat2]


#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

name = "Decision Tree"
cv_name = "K-Folds CV"
cv_name_save = "CV"
n_splits = 10
seed = 7
save_name = name+" using "+str(n_splits)+"-Folds CV"


#kfold = StratifiedKFold(n_splits=n_splits, random_state=seed) # Define the split 
kfold = KFold(n_splits=n_splits, random_state=seed) # Define the split 


#criterion='entropy', default='gini'
dtree = DecisionTreeClassifier(max_depth=3, max_features=1)

#results of arrays each of 5 elements (5 folds cv) 
results = cross_validate(dtree, X, y, cv=kfold, scoring='accuracy', return_estimator=True)

print("\n**Results**")
print("Model: {}, Cross Validation: {} {} {}{}, Number splits: {}".format(name, cv_name, "(K =", n_splits, ")", n_splits))

print('Score: {:.4f}'.format(mean(results['test_score'])))


#%%
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dtree.fit(X,y)
#class_names = ascending order
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=X.columns, class_names=['no_need_help','need_help'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#%% [markdown]
# **Distribution Graphs**

#%%
df_Level1_rank = df_Level1_no0
df_Level4_rank = df_Level4_no0


#%%
# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
fig.suptitle('Important Feature Distribution\n')


sns.distplot(df_Level1_rank['House_median'], kde=False, color="r", ax=axes[0,0], bins=25).set_title('% Level1: House median')
axes[0,0].set_ylabel('Frequency')
#axes[0,0].set(ylim=(0, 25))
axes[0,0].set(xlim=(0, 7000000))


sns.distplot(df_Level4_rank['House_median'], kde=False, color="g", ax=axes[0,1], bins=25).set_title('% Level4: House median')
axes[0,1].set_ylabel('Frequency')
#axes[0,1].set(ylim=(0, 25))
axes[0,1].set(xlim=(0, 7000000))

#df_Level1_rank['Pct_Avg_Parent Education_Graduate school/Post graduate']
sns.distplot(getNonzeroDF(df_Level1_rank, 'Pct_Avg_Parent Education_Graduate school/Post graduate'), kde=False, color="r", ax=axes[1,0], bins=25).set_title('% Level1: Pct_Parent Education_Graduate school/Post graduate')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set(ylim=(0, 25))
axes[1,0].set(xlim=(0, 100))

sns.distplot(getNonzeroDF(df_Level4_rank, 'Pct_Avg_Parent Education_Graduate school/Post graduate'), kde=False, color="g", ax=axes[1,1], bins=25).set_title('% Level4: Pct_Parent Education_Graduate school/Post graduate')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set(ylim=(0, 25))
axes[1,1].set(xlim=(0, 100))

# sns.distplot(df_Level1_rank['Pct_Avg_Parent Education_No_College'], kde=False, color="r", ax=axes[2,0], bins=25).set_title('% Level1: Pct_Parent Education_No_College')
# axes[2,0].set_ylabel('Frequency')
# sns.distplot(df_Level4_rank['Pct_Avg_Parent Education_No_College'], kde=False, color="g", ax=axes[2,1], bins=25).set_title('% Level4: Pct_Parent Education_No_College')
# axes[2,1].set_ylabel('Frequency')

sns.distplot(getNonzeroDF(df_Level1_rank, 'Pct_Avg_Parent Education_Not a high school graduate'), kde=False, color="r", ax=axes[2,0], bins=25).set_title('% Level1: Pct_Avg_Parent Education_Not a high school graduate')
axes[2,0].set_ylabel('Frequency')
axes[2,0].set(ylim=(0, 30))
axes[2,0].set(xlim=(0, 100))
sns.distplot(getNonzeroDF(df_Level4_rank, 'Pct_Avg_Parent Education_Not a high school graduate'), kde=False, color="g", ax=axes[2,1], bins=25).set_title('% Level4: Pct_Avg_Parent Education_Not a high school graduate')
axes[2,1].set_ylabel('Frequency')
axes[2,1].set(ylim=(0, 30))
axes[2,1].set(xlim=(0, 100))

sns.distplot(getNonzeroDF(df_Level1_rank, 'Pct_Avg_Multi_Ethnicity_Hispanic+Black'), kde=False, color="r", ax=axes[3,0], bins=25).set_title('%Level 1: Pct_Ethnicity_Hispanic+Black')
axes[3,0].set_ylabel('Frequency')
axes[3,0].set(ylim=(0, 30))
axes[3,0].set(xlim=(0, 100))
sns.distplot(getNonzeroDF(df_Level4_rank, 'Pct_Avg_Multi_Ethnicity_Hispanic+Black'), kde=False, color="g", ax=axes[3,1], bins=25).set_title('% Level 4: Pct_Ethnicity_Hispanic+Black')
axes[3,1].set_ylabel('Frequency')
axes[3,1].set(ylim=(0, 30))
axes[3,1].set(xlim=(0, 100))

sns.distplot(getNonzeroDF(df_Level1_rank, 'Pct_Avg_Economic Status_Economically disadvantaged'), kde=False, color="r", ax=axes[4,0], bins=25).set_title('% Level1: Pct_Economically disadvantaged')
axes[4,0].set_ylabel('Frequency')
axes[4,0].set(ylim=(0, 40))
axes[4,0].set(xlim=(0, 100))

sns.distplot(getNonzeroDF(df_Level4_rank, 'Pct_Avg_Economic Status_Economically disadvantaged'), kde=False, color="g", ax=axes[4,1], bins=25).set_title('% Level4: Pct_Economically disadvantaged')
axes[4,1].set_ylabel('Frequency')
axes[4,1].set(ylim=(0, 40))
axes[4,1].set(xlim=(0, 100))


plt.tight_layout(rect=[0, 0.03, 1, 0.95])


#%%
df_Level1 = df_Level1[attr_important_features_inter]
corr = df_Level1.corr()
top_corr_features = corr.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df_schools[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#%% [markdown]
# ### Race Percentage Distribution 
# 
# Ethnicity distribution the ``Percentage Standard Exceeded`` (Top-Tier Group)

#%%
def getDF(df, category, student_group):
    local_df = df.loc[(df['Category'] == category) & (df['Student Groups'] == student_group)]
    return local_df

df_asian = getDF(final_data, 'Ethnicity', 'Asian')
df_white = getDF(final_data, 'Ethnicity', 'White')
df_black = getDF(final_data, 'Ethnicity', 'Black or African American')
df_hispanic = getDF(final_data, 'Ethnicity', 'Hispanic or Latino')


#%%
# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
fig.suptitle('Ethnicity distribution the Percentage Standard Exceeded (Top-Tier Group)\n')

# Graph for Asian
sns.distplot(df_asian['Percentage Standard Exceeded'], kde=False, color="g", ax=axes[0,0], bins=25).set_title('% Asian Distribution')
axes[0,0].set_ylabel('Student Number Count')

# Graph for White
sns.distplot(df_white['Percentage Standard Exceeded'], kde=False, color="r", ax=axes[0,1], bins=25).set_title('% White Distribution')
axes[0,1].set_ylabel('Student Number Count')

# Graph for Black
sns.distplot(df_black['Percentage Standard Exceeded'], kde=False, color="b", ax=axes[1,0], bins=25).set_title('% Black Distribution')
axes[1,0].set_ylabel('Student Number Count')

# Graph for Hispanic
sns.distplot(df_hispanic['Percentage Standard Exceeded'], kde=False, color="c", ax=axes[1,1], bins=25).set_title('% Hispanic Distribution')
axes[1,1].set_ylabel('Student Number Count')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

==> Asian students are in the diverse range of the percentage of high scores. In short, many of the Asian students exceeded in some schools but a small portion of Asian students exceeded in other schools.

However, there are a few Black and Hispanic students who achieve the exceeded standard scores. As you can see, the graphs in the Black and Hispanic distribution, the graph bar is skewed to the left. This means that a small portion of Black and Hispanic students exceeded some other schools, but there is almost no counts that the majority or a high portion of those Black and Hispanic students achieve the high performances.
#%%
sns.distplot(df_asian['Percentage Standard Exceeded'], color="g", kde=False, label="Asian Distribution").set_title('Ethnicity distribution the Percentage Standard Exceeded (Top-Tier Group)')
sns.distplot(df_white['Percentage Standard Exceeded'], color="r", kde=False, label="White Distribution")
sns.distplot(df_hispanic['Percentage Standard Exceeded'], color="c", kde=False, label = "Hispanic Distribution")
sns.distplot(df_black['Percentage Standard Exceeded'], color="b", kde=False, label="Black Distribution")

plt.legend()
plt.show()

#plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#%% [markdown]
# Ethnicity distribution the `Percentage Standard Not Met` (Bottom-Tier Group)'

#%%
sns.distplot(df_asian['Percentage Standard Not Met'], color="g", kde=False, label="Asian Distribution").set_title('Ethnicity distribution the Percentage Standard Not Met (Bottom-Tier Group)')
sns.distplot(df_white['Percentage Standard Not Met'], color="r", kde=False, label="White Distribution")
sns.distplot(df_hispanic['Percentage Standard Not Met'], color="c", kde=False, label = "Hispanic Distribution")
sns.distplot(df_black['Percentage Standard Not Met'], color="b", kde=False, label="Black Distribution")

plt.legend()
plt.show()

#plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#%% [markdown]
# ## Further Reference
#%% [markdown]
# ### ANOVA
# 
# The one-way analysis of variance (ANOVA) is used to determine whether there are any statistically significant differences between the means of three or more independent (unrelated) groups. 
# https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/anova/

#%%
from scipy import stats

columns_to_analyze = []
for column in columns_to_analyze:
    grouped_dfs = []
    for group in df.groupby(column).groups:
        grouped_df = df.groupby(column).get_group(group)
        #reference below code
        grouped_df = grouped_df.reset_index()['Percentage Standard Exceeded']
        grouped_dfs.append(list(grouped_df.dropna()))
    F, p = stats.f_oneway(*grouped_dfs)
    print(f'{column}: {p: 0.2e}')


#%%
specific_grouped_df = df.groupby(["Category", "Student Groups"]).get_group(('Ethnicity','Asian'))
specific_grouped_df.head()


#%%
#grouped_df.reset_index()
#reset_index(): Reset the index of the DataFrame, and use the default one instead. If the DataFrame has a MultiIndex, this method can remove one or more levels.
specific_grouped_df = specific_grouped_df.reset_index()['School Code']

specific_grouped_df.head()

#%% [markdown]
# ### Chi square testing

#%%
# chi-squared test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2
# contingency table
table = [	[10, 20, 30],
			[6,  9,  17]]
print(table)
stat, p, dof, expected = chi2_contingency(table)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

#%% [markdown]
# ### Types of Statistical Data
# 
# NUMERICAL, CATEGORICAL, AND ORDINAL  
# 
# - Categorical : Ex) blonde, brown, brunette, red, etc.
# - Ordinal : Similar to categories but has clear ordering (low, medium, high)
# - Interval : If these categories were equally spaced, then the variable would be an interval variable.
# 
# https://www.dummies.com/education/math/statistics/types-of-statistical-data-numerical-categorical-and-ordinal/
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-interval-variables/
# 
# https://www.statisticshowto.datasciencecentral.com/ranked-variable/

#%%
#data (DataFrame)
#%store final_data
#wrangled data for EDA (DataFrame)
#%store df_schools
#group student info. (DataFrame)
#%store studentGroup_types
#feature selections: all (list)
get_ipython().run_line_magic('store', 'attr_important_features')
#feature selections: intersection (list)
get_ipython().run_line_magic('store', 'attr_important_features_inter')


#%%
df_schools.to_csv("df_schools.csv", sep='\t', encoding='utf-8')

df_attr_important_features = pd.DataFrame(attr_important_features, columns=["feature"])
df_attr_important_features.to_csv("attr_important_features.csv", sep='\t', encoding='utf-8')

df_attr_important_features_inter = pd.DataFrame(attr_important_features_inter, columns=["feature_inter"])
df_attr_important_features_inter.to_csv("attr_important_features_inter.csv", sep='\t', encoding='utf-8')


