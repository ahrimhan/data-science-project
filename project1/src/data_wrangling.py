#%% [markdown]
# # 1. Data Wrangling
# 
# In this notebook, we perform data cleaning, fix missing values, and add new columns with meaning values.
#%% [markdown]
# **Table of Contents**
# 
# [Data Loading and Manipulating](#Data-Loading-and-Manipulating)  
# * [CAASPP Test Scores](#CAASPP-Test-Scores)
# * [House Prices](#House-Prices)
# 
# [Joining Multiple Datasets and Cleaning Data](#Joining-Multiple-Datasets-and-Cleaning-Data)  
# [Detecting and Imputing Missing Values](#Detecting-and-Imputing-Missing-Values)
#%% [markdown]
# ## Loading Modules

#%%
import pandas as pd
import numpy as np

#To find the file encoding type
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob

#plotly
#import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf
#plotly.tools.set_credentials_file(username='ahrimhan', api_key='iGryT0mF9uXxVgbnCgT9')

#%% [markdown]
# ## Data Loading and Manipulating
#%% [markdown]
# ### CAASPP Test Scores
# We acquired the test score data for the [California Assessment of Student Performance and Progress (CAASPP)](https://caaspp.cde.ca.gov/). The test type is the Smarter Balanced English Language Arts/Literacy and Mathematics.
# 
# * The public data is available between 2015 and 2018 (4 years):
# [Research Files for Smarter Balanced Assessments](https://caaspp.cde.ca.gov/sb2018/ResearchFileList)
# 
# * The data set is too large to commit to GitHub, please refer the data set in the following link:
# [CAASPP test scores](https://drive.google.com/drive/folders/1puqmaVznmecKan-O6VNrqZy11MLeLzHo?usp=sharing)
#%% [markdown]
# #### Test data
# 
# We first load the 2018 test data.

#%%
pd.set_option('display.max_columns', None)


#%%
missing_values = ['n/a', 'na', '-', '*']


#%%
dtype_= {'Percentage Standard Exceeded': np.float64,
        'Percentage Standard Met': np.float64,
        'Percentage Standard Met and Above': np.float64,
        'Percentage Standard Nearly Met': np.float64,
        'Percentage Standard Not Met': np.float64,
        'Area 1 Percentage Above Standard': np.float64,
        'Area 1 Percentage Near Standard': np.float64,
        'Area 1 Percentage Below Standard': np.float64,
        'Area 2 Percentage Above Standard': np.float64,
        'Area 2 Percentage Near Standard': np.float64,
        'Area 2 Percentage Below Standard': np.float64,
        'Area 3 Percentage Above Standard': np.float64,
        'Area 3 Percentage Near Standard': np.float64,
        'Area 3 Percentage Below Standard': np.float64,
        'Area 4 Percentage Above Standard': np.float64,
        'Area 4 Percentage Near Standard': np.float64,
        'Area 4 Percentage Below Standard': np.float64}


#%%
df = pd.read_csv("../../CAASPP/all/sb_ca2018_all.csv", na_values = missing_values, dtype=dtype_)


#%%
#time series plot
path = "../../CAASPP/all/"
allFiles = glob.glob(path + "/*.csv")

list_ = []

for file_ in allFiles:
    df_temp = pd.read_csv(file_, na_values = missing_values, dtype=dtype_, index_col=None, header=0)
    list_.append(df_temp)

df_all = pd.concat(list_, axis = 0, ignore_index = True)


#%%
df.shape


#%%
df.head()


#%%
df.info()

#%% [markdown]
# #### Entity tables
# The following entity files list the County, District, and School entity names and codes for all entities as the existed in the administration year selected. This file must be merged with the test data file to join these entity names with the appropriate score data.

#%%
#find the file encoding type (warning: it takes so long time, so just put the exact number)
#with open("../../CAASPP/entities/sb_ca2018entities.csv", 'rb') as f:
#    result = chardet.detect(f.read())  # or readline if the file is large
    
result = {'encoding': 'Windows-1252', 'confidence': 0.7299741290274674, 'language': ''}


#%%
entities = pd.read_csv("../../CAASPP/entities/sb_ca2018entities.csv", encoding=result['encoding'], na_values = missing_values)


#%%
entities.shape


#%%
entities.info()


#%%
entities.drop(columns='Filler', inplace=True)

#%% [markdown]
# The `Zip Code` is type string. For further merging cases, we change the `Zip Code` to the integer type to maintain the same type.

#%%
#tip: we need same data types of keywords for merging

#missing values

#convert 'Zip Code' column type: string to int64
entities['Zip Code'] = pd.to_numeric(entities['Zip Code'], errors='coerce').fillna(0).astype(np.int64)


#%%
entities.head()

#%% [markdown]
# #### Subgroup and Test ID tables
# Each [`Subgroup ID`](https://caaspp.cde.ca.gov/sb2018/research_fixfileformat18) has the following meanings. We can investigate the characteristics of individual students. 

#%%
subgroup = pd.read_csv("../../CAASPP/Subgroups.txt", header=None, na_values = missing_values)
subgroup.shape


#%%
subgroup.head()

#%% [markdown]
# We clean the `Subgroup ID` table. The first and the second columns are redundant, so the first one is dropped. We name the columns. 
# 
# The category "English-Language Fluency" has the 8 student groups:
# * Fluent English proficient and English only
# * Initial fluent English proficient (IFEP)
# * Reclassified fluent English proficient (RFEP)
# * English learners (ELs) enrolled in school in the U.S. fewer than 12 months
# * English learners enrolled in school in the U.S. 12 months or more
# * English learner
# * Ever-ELs
# * English only
# 
# Those groups are related as follows:
# * Total students = Fluent English proficient and English only + English learner
# * Fluent English proficient and English only = Initial fluent English proficient (IFEP) + Reclassified fluent English proficient (RFEP) + English only
# * English learner = English learners (ELs) enrolled in school in the U.S. fewer than 12 months + English learners enrolled in school in the U.S. 12 months or more
# * Ever-ELs = Reclassified fluent English proficient (RFEP) + English learner
# 
# The definition of **Initial fluent English proficient (IFEP)** is as follows:
# On the first ever taken California English Language Development Test (CELDT), if your child scored at the Early Advanced or Advanced level of language proficiency, your child is identified as "Initially Fluent English Proficient" or IFEP, meaning that your child has enough language proficiency to continue learning like native language speaking and doesn’t need additional English Language Development support [\[1\]](https://www.cde.ca.gov/ta/tg/ep/elpacipld.asp), [\[2\]](https://stoneridge.rcsdk8.org/post/english-learner-el).

#%%
#delete first column (redundant with the second column); axis = 0 (index) and axis =1 (column), inplace=True means adjusting
subgroup.drop(0, axis=1, inplace=True)
subgroup.columns = ['Subgroup ID', 'Student Groups', 'Category']
subgroup.sort_values("Category")


#%%
subgroup.head()


#%%
#Strip whitespaces (including newlines) or a set of specified characters from each string
subgroup['Category'] = subgroup['Category'].map(lambda x: x.replace('"', '').strip())
subgroup['Student Groups'] = subgroup['Student Groups'].map(lambda x: x.replace('"', '').strip())

#%% [markdown]
# The `Test ID` has the following meanings. The `Test ID` is 1-4; 1 represents ELA and 2 represents mathematics, respectively. We do not consider 3 and 4 because they are CAA (California Alternative Assessments) scores. The CAA scores are taken by students in grades 3–8 and grade 11 whose individualized education program (IEP) teams have determined that the student's cognitive disabilities prevent him or her from taking the online CAASPP Smarter Balanced assessments.
#%% [markdown]
# For readability, we convert the **column type** of `Test Id` from `int64` to `string`.

#%%
tests_id = pd.read_csv("../../CAASPP/Tests.txt", header=None, na_values = missing_values)
tests_id


#%%
#performance better (ver 1).

# tests_id.columns = ['Test Id Name', 'Test Id', 'Test Name']
# tests_id.drop(0, axis=0, inplace=True)

# tests_id['Test Id Name'] = tests_id['Test Id Name'].replace("1", "English")
# tests_id['Test Id Name'] = tests_id['Test Id Name'].replace("2", "Mathematics")
# tests_id['Test Id Name'] = tests_id['Test Id Name'].replace("3", "CAA-English")                                                        
# tests_id['Test Id Name'] = tests_id['Test Id Name'].replace("4", "CAA-Mathematics") 

# #type conversion : string to int64
# tests_id['Test Id'] = pd.to_numeric(tests_id['Test Id'], errors='coerce').fillna(0).astype(np.int64)
# tests_id.drop(columns='Test Name', inplace=True)

# tests_id

# df = pd.merge(tests_id, df, how='inner', on=['Test Id'])
# df.drop(columns='Test Id', inplace=True)
# df.rename(columns={'Test Id Name': 'Test Id'}, inplace=True)


#%%
#performance better (ver 2).
#convert 'Test Id' column type: int64 to string

df['Test Id'] = df['Test Id'].replace(1, "English")
df['Test Id'] = df['Test Id'].replace(2, "Mathematics")
df['Test Id'] = df['Test Id'].replace(3, "CAA-English")
df['Test Id'] = df['Test Id'].replace(4, "CAA-Mathematics")

#%% [markdown]
# I decided to use only the next columns: ‘Country Code’, ‘District Code’, ‘School Code’, ‘Test Year’, ‘Subgroup ID’, ‘Grade’, ‘Test Id’, ‘Students with Scores’, and achievement levels. The [minimum and maximum test scale score ranges](https://caaspp.cde.ca.gov/sb2016/ScaleScoreRanges) are provided, and the ‘Mean Scale Score’ is used to determine four achievement levels: ‘Percentage Standard Exceeded’ ‘Percentage Standard Met’, ‘Percentage Standard Nearly Met’, ‘Percentage Standard Not Met’. Many studies showed that discretization can lead to improved predictive accuracy and is more understandable. The test score data also has [area descriptors](https://caaspp.cde.ca.gov/sb2018/UnderstandingCAASPPReports). There are 4 areas of reading, writing, listening, and research/inquiry for ELA whereas 3 areas of concepts and procedures, problem solving/modeling and data analysis, and communicating reasoning for mathematics. For each area, the achievement levels are divided into ‘Above Standard’, ‘Near Standard’, and ‘Below Standard’ depending on the scale scores compared to the ‘Standard Met’ achievement level.

#%%
#Percentage Standard Met and Above = Percentage Standard Exceeded + Percentage Standard Met
df.drop(columns=['Filler', 'Total Tested At Entity Level', 'Total Tested with Scores', 'CAASPP Reported Enrollment', 'Students Tested', 'Mean Scale Score', 'Percentage Standard Met and Above'], inplace=True)

#%% [markdown]
# ### House Prices
# 
# * [Zillow research data](https://www.zillow.com/research/CAASPP/): House prices based on zipcodes
# 
# The Zillow Home Value Index (ZHVI) data was imported and loaded. The ZHVI is a seasonally adjusted measure of the median estimated home value across a given region and housing type. The data was collected from April 1996 to November 2018 on monthly basis. 
# 
# The column name `RegionName` denotes zipcode so it is renamed as `Zip Code`. The `Zip Code` is set as the index.

#%%
result= {'encoding': 'ISO-8859-1', 'confidence': 0.73, 'language': ''}

#%% [markdown]
# **Loading ver 1.**
# Data manipulation using DatetimeIndex objects.

#%%
#tip: column names has to be changed before setting an index; if setting index while reading csv file, error!
df_house_price2 = pd.read_csv("../../CAASPP/house/Zip_Zhvi_AllHomes.csv", encoding=result['encoding'], na_values = missing_values)

df_house_price2.rename(columns={'RegionName': 'Zip Code'}, inplace=True)
df_house_price2.set_index('Zip Code', inplace=True)
df_house_price2.head()


#%%
df_house_price2.shape


#%%
df_house_price2.info()


#%%
df_house_price2.drop(columns=['RegionID', 'City', 'State', 'Metro', 'CountyName', 'SizeRank'], inplace=True)

#%% [markdown]
# We make columns as **DatetimeIndex objects**. I found this is more convenient and safe for dealing with time related data.

#%%
#tip: when dealing with time data, it is much better to use time related libraries!
#pandas.DatetimeIndex
df_house_price2.columns = pd.to_datetime(df_house_price2.columns)

#%% [markdown]
# We analyze the test scores for the years of 2015 to 2018. 
# I cleaned up the data by dropping house prices that are less than 2015 or greater than 2018.
# To analyze the school performance on a yearly basis, the monthly prices were grouped by each year into a median value.

#%%
#clean data - remain data from years of 2015 to 2018
dropColumns = [ x for x in df_house_price2.columns
                if (x.year < 2015 or x.year > 2018) ]

df_house_price2.drop(columns=dropColumns, inplace=True)


#%%
df_house_price2.columns

#%% [markdown]
# **House median prices**

#%%
df_house_price_grouped_median = df_house_price2.groupby(pd.Grouper(freq='Y', axis=1), axis=1).median()
df_house_price_grouped_median.head()


#%%
#clean data
#convert columns from DatetimeIndex to int64 for compatibility

df_house_price_grouped_median.columns = [x.year for x in df_house_price_grouped_median.columns]


#%%
df_house_price_stacked2 = df_house_price_grouped_median.stack().to_frame()
df_house_price_stacked2.columns = ['House_median']
df_house_price_stacked2.index.names = ['Zip Code', 'Test Year']
df_house_price_stacked2.head(10)

#%% [markdown]
# **Loading ver 2.**
# Data manipulation using user-defined functions.

#%%
#RegionName = zipcode
df_house_price = pd.read_csv("../../CAASPP/house/Zip_Zhvi_AllHomes.csv", encoding=result['encoding'], na_values = missing_values)
df_house_price.rename(columns={'RegionName': 'Zip Code'}, inplace=True)
df_house_price.set_index('Zip Code', inplace=True)


#%%
def getYearPart(year_month):
    res = year_month.split('-')
    #if there is no '-', just return its original value
    return res[0]
def getYearPartInt(year_month):
    res = getYearPart(year_month)
    if res.isdigit():
        return int(res)
    return res


#%%
#consider years of 2015 to 2018 (4 years)

dropColumns = [ x for x in df_house_price.columns
                if (not getYearPart(x).isdigit()) or (int(getYearPart(x)) < 2015 or int(getYearPart(x)) > 2018) ]
df_house_price.drop(columns=dropColumns, inplace=True)


#%%
df_house_price_grouped = df_house_price.groupby(getYearPartInt, axis=1).median()


#%%
df_house_price_stacked = df_house_price_grouped.stack().to_frame()
df_house_price_stacked.columns = ['House_median']
df_house_price_stacked.index.names = ['Zip Code', 'Test Year']


#%%
print (df_house_price_stacked.index.get_level_values(0).dtype)
print (df_house_price_stacked.index.get_level_values(1).dtype)


#%%
df_house_price_stacked.loc[(60657, 2018)]


#%%
del df_house_price_stacked

#%% [markdown]
# ### Additional Datasets
# Additional datasets are obtained in the following sites:
# * [Civil Rights Data Collection](https://ocrdata.ed.gov/): Teacher demographics
# 
# * [GreatSchools API](https://www.greatschools.org/api/docs/technical-overview/): School profile, school reviews, school censuc data, nearby schools
#%% [markdown]
# ## Joining Multiple Datasets and Cleaning Data
# There are multiple dataset and we need to merge efficiently to obtain useful and clean data. 
#%% [markdown]
# **1. Select all grades (`Grade` == 13) and "Smart Balanced (basic official)" test type (`Test Type` == B).**
#%% [markdown]
# [The `Grade 13` denotes all grades](https://caaspp.cde.ca.gov/sb2018/research_fixfileformat18), so we decided to use data only 13 for minimum sample size. I believe the aggregated data at each school level is enough for representing the characteristics of public schools in California. 
# 
# All the test scores are from the *Smarter Balanced English Language Arts/Literacy and Mathematics* (`Test Type` = 'B')

#%%
df = df.loc[(df['Grade'] == 13) & (df['Test Type'] == "B"), :]

#%% [markdown]
# We dropped the columns `Test Type` and `Grade`. Those columns do not convey any important information anymore.

#%%
dropColumns_entity = ['Test Type', 'Grade']
df.drop(columns=dropColumns_entity, inplace=True)

#%% [markdown]
# **2. Merge the entity table.**
# 
# We append the specific names to the test score DataFrame by merging two tables (Test data + entities).

#%%
allGradesDf_entity = pd.merge(entities, df, how='inner', on=['School Code', 'District Code', 'County Code', 'Test Year'])

#%% [markdown]
# **3. Merge the house price data.**
# 
# Now, we merge the house prices and test score data.

#%%
allGradesDf_entity_house = pd.merge(allGradesDf_entity, df_house_price_stacked2, how='inner', 
                  left_on=['Zip Code', 'Test Year'], right_index=True)

#%% [markdown]
# **4. merge `Subgroup ID` table.**

#%%
allGradesDf_entity_house = pd.merge(subgroup, allGradesDf_entity_house, on=['Subgroup ID'])

#%% [markdown]
# ## Detecting and Imputing Missing Values

#%%
allGradesDf_entity_house.info()

#%% [markdown]
# If we chain a `.sum()` method on, we’re given a list of all the summations of each column regarding `missing values`. We notice that only score columns have the missing values, and the number of missing values for every score column are same. This means that one row has all missing scores or all scores.

#%%
allGradesDf_entity_house.isnull().sum()


#%%
allGradesDf_entity_house.isnull().values.any()

#%% [markdown]
# In order to get the total summation of all missing values in the DataFrame, we chain two .sum() methods together.  
# https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/

#%%
allGradesDf_entity_house.isnull().sum().sum()


#%%
null_data = allGradesDf_entity_house[allGradesDf_entity_house.isnull().any(axis=1)]
null_data.head()

#%% [markdown]
# **We finally obtain the clean data.**

#%%
#Drop missing observations
final_data = allGradesDf_entity_house.dropna()


#%%
# Drop school data containing "Program" that are not official schools
# Example: "Irvine Unified District Level Program" or "Alternative Education-San Joaquin High"
excludeSchoolNames = ["Program", "Alternative"]
final_data = final_data[~final_data["School Name"].str.contains('|'.join(excludeSchoolNames), case=False, na=False)]


#%%
final_data.to_csv("final_school_data.csv", sep='\t', encoding='utf-8')


