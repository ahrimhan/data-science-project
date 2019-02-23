#%% [markdown]
# # 2. Data Visualization
#%% [markdown]
# **Data Loading**
# 
# The data has been cleand in data wrangling notebook.  
# Please refer to [`data_wrangling.ipynb`](https://github.com/ahrimhan/data-science-project/blob/master/project1/data_wrangling.ipynb) file.  
# Here we load the saved data `final_school_data.csv`. This data set is too large to commit to GitHub, and you can download this file in the following link:  
# [CAASPP test scores](https://drive.google.com/drive/folders/1puqmaVznmecKan-O6VNrqZy11MLeLzHo?usp=sharing).
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


#%%
pd.set_option('display.max_columns', None)

#%%
df = pd.read_csv('final_school_data.csv', sep='\t', encoding='utf-8')
final_data = df
final_data.head()
final_data.drop("Unnamed: 0", axis=1, inplace=True)

#%%
final_data.head()

#%%
final_data.info()

#%% [markdown]
# **Aggregation (School data to District and County levels)**
# 
# The score data at a county or a district level is not available. Thus, school data is grouped into the county or the district level. We only select the school data (`Type Id` == 7).
# 
# * **`Type Id` : Type of scores**
# 
#     * Type Id
#     (‘04’ = State,
#     ‘05’ = County,
#     ‘06’ = District,
#     ‘07’ = School,
#     ‘09’ = Direct Funded Charter School,
#     ‘10’ = Locally Funded Charter School)

#%%
final_data.groupby("Type Id").count()


#%%
#school-level data

#for-later-use
final_data_school_all = final_data

final_data_school = final_data.loc[(final_data["Type Id"] == 7)]
final_data_school = final_data_school.round(2)
final_data_school['House_median'] = final_data_school['House_median'].astype('int64')
final_data = final_data_school

print("School Name")
print(final_data_school['School Name'].nunique())


#%%
#county-level data
final_data_county = final_data.groupby(["County Name", "Student Groups", "Category", "Test Id"], as_index=False).mean()

final_data_county = final_data_county.round(2)
final_data_county['House_median'] = final_data_county['House_median'].astype('int64')

print(final_data_county.columns[0])
print(final_data_county['County Name'].nunique())


#%%
#district-level data
final_data_district = final_data.groupby(["District Name", "Student Groups", "Category", "Test Id"], as_index=False).mean()

final_data_district = final_data_district.round(2)
final_data_district['House_median'] = final_data_district['House_median'].astype('int64')

print(final_data_district.columns[0])
print(final_data_district["District Name"].nunique())

#%% [markdown]
# **Hypothesis**  
# 
# We first start with the following hypothesis.  
# - The schools with many Asian students tend to achieve high scores.
# - The schools with high-income families tend to achieve high scores.
# - The schools with highly educated parents tend to achieve high scores.
# - The schools surrounded by high house costs tend to achieve high scores.
#%% [markdown]
# **Type of graphs**
# 
# For utilizing advandced features, we used the [**seaborn**](https://seaborn.pydata.org/index.html), [**Bokeh**](http://bokeh.pydata.org/en/latest/), and [**Plotly**](https://plot.ly/python/) libraries (To Be Developed: TBD).  
# The data was explored to find trends, insights, and potential outliers. 
# In this California score data, there are 52 counties, 784 districts, and 6,539 schools.
# 
# * **Comparison (Bar plot)**
# 
# * **Correlations (Scatterplot)**
#     
# * **Distribution (Histogram)** - TBD
# 
# * **Time-Series plot**
#%% [markdown]
# ## 2.1 Comparison (Bar plot)  - Category
# 
# **How do students are different in achievement levels for each category?**   
# 
# We provide two different version of bar plots for each category. The data is **grouped by counties**:
# * all four achievement levels in a stacked bar
# * specific achievement levels in a parallelized bar
#     
# ### [Results]
# 
# * **Gender:**  
#     **Female students exceed male students in English, while male students exceed female students in Mathematics.**  
#     - In the English subject at the "Standard Exceeded" level, females students are 6.4% more than males students.
#     - In the mathematics subject at the "Standard Exceeded" level, males students are 1.4% more than female students.
#     - At the "Standard Met" above level ("Standard Exceeded" + "Standard Met"), females students are 10.8% more than males students in English (females: 50.9% > males: 40.1%). In mathematics, there are not much difference (males: 34.8% > females: 34.6%).
#     - At the "Standard Met" above level , female students are 16.3% more in English than in mathematics. **The subject difference of female students is much higher than male students.** The males students are 5.3% more in English than in mathematics.  
# 
# 
# * **Ethnicity:**  
#     **In both English and mathematics, Asian students achieve the best performance, while Black or African American and American Indian or Alaska Native students achieve the lowest performance.** 
#     - In both English and mathematics, **students' achievements are higher (there are the most "Standard Exceeded" students) in the order of Asian, Filipino, two or more races, and white.**
#     - In both English and mathematics, **students' achievements are lower (there are the most "Standard Not Met" students) in the order of Black or African American, American Indian or Alaska Native, Native Hawaiian or Pacific Islander, and Hispanic or Latino.**
#     -  **The ethnic group of students in the "Standard Not Met" level have much more difficulties in mathematics than English.**  
# 
#         
# * **English-Language Fluency:**  
#     **In both English and mathematics, Initial Fluent English Proficient (IFEP) students achieve the best performance.**   
#     - In California, students whose home language is not English are required by law to be assessed in English language proficiency. Thus, the IFEP students have enough language proficiency or are native language speakers, and their parents may have moved from other countries and are immgirants. **This is very interesting insights that IFEP students highly exceed English only students in both English and mathematics.** The percentage of standard exceeded students of IFEP are 38.2% (English) and 33.1% (mathematics), while those of English only are 20.9% (English) and 15.9 (mathematics). I could observe that this trend becomes more obvious in the districts where many Asian immigrants live. From this result, **I can insist that immigrants have high educational interests and efforts.**  
# 
# 
# * **Economic Status:**  
#     **Economically disadvantaged students have much more difficulties than not-economically disadvantaged students.**  
#     - **Almost half of the economically disadvantaged students are NOT standard met in mathematics.** For example, 45.4% of economically disadvantaged students are "Standard Not Met" in mathematics and 37.4% are "Standard Not Met" in English.  
#     
#     
# * **Disability Status:**  
#     **A small number of students with disabilities can achieve the best performance.** (English: 4.6%, mathematics: 4.5%).
#     - The majority of students with disabilitie are in the "Standard Not Met" level (English: 66.7%, mathematics: 71.1%).
#     - As in other disadvantaged or minor groups, the students with disability have more difficulties in mathematics.  
#     
#     
# * **Parent Education:**  
#     **The higher the level of parental education, the higher the achievement of students.**
#     - The graphs apparently show that **students' achievements are higher in the order of the parents' education of "graduate school/post graduate", "college graduate", "some college (includes AA degree)" , "high school graduate", and "not a high school graduate".**
#%% [markdown]
# The student groups in each category are as follows.

#%%
#Student Groups (available in final_data)
final_data_school.groupby(["Category", "Student Groups"]).count()


#%%
#finalize the data for visualization
#default: final_data = final_data_school
final_data = final_data_county


#%%
# all four achievement levels in a stacked bar
def stackedbar(df, category):
    scoreLevel = re.sub('Name', '', df.columns[0])
    
    df_local=df.loc[df['Category'] == category]
    #graph_columns = ['Category','Student Groups','Test Id', 'Percentage Standard Exceeded','Percentage Standard Met', 'Percentage Standard Nearly Met','Percentage Standard Not Met']
    graph_columns = ['Category','Student Groups','Test Id', 'Percentage Standard Not Met','Percentage Standard Nearly Met', 'Percentage Standard Met', 'Percentage Standard Exceeded']
    
    x= df_local[graph_columns]
    y=x.set_index(['Category', 'Student Groups', 'Test Id'])
    z=y.groupby(['Student Groups', 'Test Id']).mean()
    #TBD - use Bokeh libraries for exact number labeling
    #print(z)
    z.plot.bar(stacked=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=90)
    plt.title(scoreLevel + ": " + category)


#%%
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf

init_notebook_mode(connected=True)
cf.go_offline()

def stackedbar_interactive(df, category):
    scoreLevel = re.sub('Name', '', df.columns[0])
    
    df_local=df.loc[df['Category'] == category]
    #graph_columns = ['Category','Student Groups','Test Id', 'Percentage Standard Exceeded','Percentage Standard Met', 'Percentage Standard Nearly Met','Percentage Standard Not Met']
    graph_columns = ['Category','Student Groups','Test Id', 'Percentage Standard Not Met','Percentage Standard Nearly Met', 'Percentage Standard Met', 'Percentage Standard Exceeded']
    
    df_local=df_local[graph_columns]
    df_local=df_local.set_index(['Student Groups', 'Test Id'])
    df_local_group_by=df_local.groupby(['Student Groups', 'Test Id']).mean()

    cf.set_config_file(offline=False, world_readable=True, theme='ggplot')
    df_local_group_by.iplot(kind='bar', barmode='stack', filename='cufflinks/stacked-bar-chart')
    
        
    


#%%
final_id_vars = ['Subgroup ID',
 'Student Groups',
 'Category',
 'County Code',
 'District Code',
 'School Code',
 'Test Year',
 'Type Id',
 'School Name', 'District Name', 'County Name',
 'Zip Code',
 'Test Id',
 'Students with Scores',
 'Area 1 Percentage Above Standard',
 'Area 1 Percentage Near Standard',
 'Area 1 Percentage Below Standard',
 'Area 2 Percentage Above Standard',
 'Area 2 Percentage Near Standard',
 'Area 2 Percentage Below Standard',
 'Area 3 Percentage Above Standard',
 'Area 3 Percentage Near Standard',
 'Area 3 Percentage Below Standard',
 'Area 4 Percentage Above Standard',
 'Area 4 Percentage Near Standard',
 'Area 4 Percentage Below Standard',
 'House_median']

final_value_vars = ['Percentage Standard Exceeded',
 'Percentage Standard Met',
 'Percentage Standard Nearly Met',
 'Percentage Standard Not Met']


#%%
final_data_ex = pd.melt(final_data_school, id_vars=final_id_vars, value_vars=final_value_vars, var_name='Performance Group', value_name='Performance Group Percentage')
final_data_ex.head()


#%%
# specific achievement levels in a parallelized bar
def drawBarplotFacetGridEx(df, category, perfCols):
    df_local = df.loc[(df['Category'] == category)]
    
    df_local = df_local.loc[df_local['Performance Group'].isin(perfCols)]
    
    g = sns.FacetGrid(df_local, col="Test Id", height=6)
    g.map(sns.barplot, "Student Groups", "Performance Group Percentage", 'Performance Group', palette="rocket", ci=None )
    g.add_legend()
    g.set_xticklabels(rotation=90)
    g.fig.suptitle(category, size=20, x = 0.4, y = 1.1)
    plt.show()

#%% [markdown]
# ### [Overall in counties]
# 
# **Gender**

#%%
stackedbar(final_data, 'Gender')


#%%
stackedbar_interactive(final_data, 'Gender')


#%%
#FacetGrid : same legend
drawBarplotFacetGridEx(final_data_ex, 'Gender', ['Percentage Standard Exceeded', 'Percentage Standard Not Met'])

#%% [markdown]
# **Ethnicity**

#%%
stackedbar(final_data, 'Ethnicity')


#%%
drawBarplotFacetGridEx(final_data_ex, 'Ethnicity', ['Percentage Standard Exceeded', 'Percentage Standard Not Met'])

#%% [markdown]
# **English-Language Fluency**

#%%
stackedbar(final_data, 'English-Language Fluency')


#%%
drawBarplotFacetGridEx(final_data_ex, 'English-Language Fluency', ['Percentage Standard Exceeded', 'Percentage Standard Not Met'])

#%% [markdown]
# **Economic Status**

#%%
stackedbar(final_data, 'Economic Status')


#%%
drawBarplotFacetGridEx(final_data_ex, 'Economic Status', ['Percentage Standard Exceeded', 'Percentage Standard Not Met'])

#%% [markdown]
# **Disability Status**

#%%
stackedbar(final_data, 'Disability Status')


#%%
drawBarplotFacetGridEx(final_data_ex, 'Disability Status', ['Percentage Standard Exceeded', 'Percentage Standard Not Met'])

#%% [markdown]
# **Parent Education**

#%%
stackedbar(final_data, 'Parent Education')


#%%
drawBarplotFacetGridEx(final_data_ex, 'Parent Education', ['Percentage Standard Exceeded', 'Percentage Standard Not Met'])

#%% [markdown]
# ## 2.2 Comparison (Bar plot)  -  Top and low performance counties
# 
# **What features can you find in the top and bottom performance groups?** 
# 
# We analyzed **5 best and worst performing counties** (58 counties x 10%).
# The results can be summarized as follows.
# * **English Top 5** County Names:
# ['Santa Clara', 'Marin', 'Placer', 'San Mateo', 'Orange']
# * **Mathematics Top 5** County Names:
# ['Santa Clara', 'Marin', 'San Mateo', 'Orange', 'Placer']
# * **English Low 5** County Names:
# ['Lake', 'Kings', 'Colusa', 'Humboldt', 'Monterey']
# * **Mathematics Low 5** County Names:
# ['Lake', 'Kings', 'Merced', 'Mendocino', 'Monterey']
# 
# ### [Results]
# 
# * The best performance counties have higher house median prices. In constrast, the worst performance conties have lower house median prices. **Thus, test performance is closely related to the economic capabilities of the family to which the student belongs.**
# 
# * We found that in the best performing counties, the percentage of white students is much higher than the percentage of white students in the whole county.
# 
# * Hispanic and Latino students are far more likely to be in the worst performing group than the best performing group. Likewise, black and American Indian students are more involved in the group with the worst results. In contrast, Asian and white students are more likely to be in the best performing group than the worst performing group.
# 
# * The English learners have more difficuties in studying both English and Mathematics than the fluent English speakers.
# 
# * When students' parents graduate from graduate schools/post graduates or colleges, students are much more likely to be in the best perfoming group. For those students, the best performing groups are much larger thatn the worst performing groups. In contrast, students are more likely to be in the worst perfoming group when their parents are high school graduates or have lower eduation.
#%% [markdown]
# ### [Characteristics in the top and bottom performance groups in counties]
#%% [markdown]
# To have a rough insight, we have drawn the graphs of the test scores (i.e., "Performance Group Percentage" of each "Performance Group").

#%%
#sns.set(style="whitegrid")
sns.set_style(style='ticks')
sns.set(rc={'figure.figsize':(20,16)})


#%%
saturation_p =0.5
asepct_p = 1.5

def drawBarplotCatplot2(df, category, saturation_p, asepct_p, _x, _y, _col):
    df_local = df.loc[(df["Category"] == category)]
    
    #x="Student Groups", y="Students with Scores", col="Test Id", hue="Rank"
    g = sns.catplot(x=_x, y=_y, col=_col,
                data=df_local, saturation=saturation_p,
                kind="bar", ci=None, aspect=asepct_p)
    (
        g.set_axis_labels(_x, _y)
        .set_titles("{col_name}")
        .despine(left=True)
        .set_xticklabels(rotation=90)
    )  

drawBarplotCatplot2(final_data_ex, 'All Students', saturation_p, asepct_p, "County Name", "Performance Group Percentage", "Performance Group")


#%%
#df: data
#scoreLevel: "School Name", "District Name", "County Name"
#num: top and bottom parameter
def getDFTop_Low(df, scoreLevel, num):
    english_df = df.loc[(df["Student Groups"] == "All Students") 
                                   & (df["Test Id"] == "English")]    
    english_top = english_df.nlargest(num, 'Percentage Standard Exceeded')
    english_low = english_df.nlargest(num, 'Percentage Standard Not Met')
    
    math_df = df.loc[(df["Student Groups"] == "All Students") 
                                   & (df["Test Id"] == "Mathematics")]
    math_top = math_df.nlargest(num, 'Percentage Standard Exceeded')
    math_low = math_df.nlargest(num, 'Percentage Standard Not Met')
    
    print("{}{}{}{}{}".format("* English Top ", num, " ", scoreLevel,"s:"))
    #print Series
    print(english_top[scoreLevel].values.tolist())
    print("{}{}{}{}{}".format("* Mathematics Top ", num, " ", scoreLevel,"s:"))
    print(math_top[scoreLevel].values.tolist())
    print("{}{}{}{}{}".format("* English Low ", num, " ", scoreLevel,"s:"))
    print(english_low[scoreLevel].values.tolist())
    print("{}{}{}{}{}".format("* Mathematics Low ", num, " ", scoreLevel,"s:"))
    print(math_low[scoreLevel].values.tolist())
    
    #Retreive all data of the top and bottom schools/districts/counties
    top_english_all = df.loc[(df[scoreLevel].isin(english_top[scoreLevel])) &
                                                  (df["Test Id"] == "English")]
    low_english_all = df.loc[(df[scoreLevel].isin(english_low[scoreLevel])) &
                                                   (df["Test Id"] == "English")]
      
    top_math_all = df.loc[(df[scoreLevel].isin(math_top[scoreLevel])) &
                                                  (df["Test Id"] == "Mathematics")]
    low_math_all = df.loc[(df[scoreLevel].isin(math_low[scoreLevel])) &
                                                   (df["Test Id"] == "Mathematics")]
    
    top_english_all["Rank"] = "Top"
    low_english_all["Rank"] = "Bottom"
    top_math_all["Rank"] = "Top"
    low_math_all["Rank"] = "Bottom"
    
    df_local = pd.concat([top_english_all, low_english_all, top_math_all, low_math_all])
    
    return df_local


#%%
dfTop_Low = getDFTop_Low(final_data_county, "County Name", 5)
dfTop_Low

#%% [markdown]
# **House Prices**

#%%
def drawBarplotCatplot1(df, category, testId, _x, _y, _hue, _col):
    df_local = df.loc[(df["Category"] == category) & (df['Test Id'] == testId)]
    g = sns.catplot(x=_x, y=_y, hue=_hue, col=_col,
                    data=df_local, saturation=saturation_p,
                    kind="bar", ci=None, aspect=asepct_p)
    (g.set_axis_labels(_x, _y)
     # .set_xticklabels(["Men", "Women", "Children"])
     #.set(ylim=(0, 1))
     .despine(left=True)
     .set_xticklabels(rotation=90))  

drawBarplotCatplot1(dfTop_Low, "All Students", "English", "County Name", "House_median", "Rank", "Test Id")
drawBarplotCatplot1(dfTop_Low, "All Students", "Mathematics", "County Name", "House_median", "Rank", "Test Id")

#drawBarplotCatplot(dfTop_Low, "All Students", saturation_p, asepct_p, "County Name", "House_median", "Test Id", "Rank")


#%%
def drawBarplotCatplot(df, category, saturation_p, asepct_p, _x, _y, _col, _hue):
    df_local = df.loc[(df["Category"] == category)]
    
    #x="Student Groups", y="Students with Scores", col="Test Id", hue="Rank"
    g = sns.catplot(x=_x, y=_y, col=_col, hue=_hue,
                data=df_local, saturation=saturation_p,
                kind="bar", ci=None, aspect=asepct_p)
    (
        g.set_axis_labels(category, "Number of Students")
        # .set_xticklabels(["Men", "Women", "Children"])
        .set_titles("{col_name} {col_var}")
        #  .set(ylim=(0, 1))
        .despine(left=True)
        .set_xticklabels(rotation=90)
    )  

#%% [markdown]
# **Gender**

#%%
drawBarplotCatplot(dfTop_Low, "Gender", saturation_p, asepct_p, "Student Groups", "Students with Scores", "Test Id", "Rank")

#%% [markdown]
# **Ethnicity**

#%%
drawBarplotCatplot(dfTop_Low, "Ethnicity", saturation_p, asepct_p, "Student Groups", "Students with Scores", "Test Id", "Rank")

#%% [markdown]
# **English-Language Fluency**

#%%
drawBarplotCatplot(dfTop_Low, "English-Language Fluency", saturation_p, asepct_p, "Student Groups", "Students with Scores", "Test Id", "Rank")

#%% [markdown]
# **Economic Status**

#%%
drawBarplotCatplot(dfTop_Low, "Economic Status", saturation_p, asepct_p, "Student Groups", "Students with Scores", "Test Id", "Rank")

#%% [markdown]
# **Parent Education**

#%%
drawBarplotCatplot(dfTop_Low, "Parent Education", saturation_p, asepct_p, "Student Groups", "Students with Scores", "Test Id", "Rank")

#%% [markdown]
# ### [Individual Top and Bottom schools in Each Category]

#%%
#List of data of the best performance students for each student groups
idx = final_data_school.groupby(["Category", "Student Groups"])["Percentage Standard Exceeded"].transform(max) == final_data_school["Percentage Standard Exceeded"]
final_data_school[idx].head()


#%%
#List of data of the lowest performance students for each student groups
idx = final_data_school.groupby(["Category", "Student Groups"])["Percentage Standard Not Met"].transform(max) == final_data_school["Percentage Standard Not Met"]
final_data_school[idx].head()

#%% [markdown]
# ## 2.3 Correlations (Scatterplot)
# 
# We provide the scatter plots between two following factors.  
# * Percentage of Standard Exceeded vs. House prices
# * Percentage of Standard Not Met vs. House prices
#    
# ### [Results]
# We could observe the strong correlations between the test scores and the house prices.
# In conclusion, students who live in areas with high housing prices have higher test scores.

#%%
def drawScatterplot(df, category, levelName, perfCol):
    df_local = df.loc[(df['Category'] == category)]
    #g = sns.FacetGrid(df_local, col="Test Id", hue="Student Groups")
    g = sns.FacetGrid(df_local, col="Test Id", hue="Student Groups", size=10)
    g.map(plt.scatter, levelName, perfCol, alpha=.7)
    g.add_legend();
    g.set_xticklabels(rotation=90) 

#drawScatterplot(final_data, 'Ethnicity', "County Name", 'Percentage Standard Not Met')


#%%
df_local = final_data.loc[(final_data['Student Groups'] == "All Students")].groupby('School Code').mean()
ax = sns.scatterplot(x="House_median", y='Percentage Standard Exceeded', data=df_local)


#%%
df_local = final_data.loc[(final_data['Student Groups'] == "All Students")].groupby('School Code').mean()
ax = sns.scatterplot(x="House_median", y='Percentage Standard Not Met', data=df_local)

#%% [markdown]
# ## 2.4 Distribution (Histogram)
# 
# TBD
#%% [markdown]
# ## 2.5 Time-Series Plot
# 
# TBD
#%% [markdown]
# ## 2.6 Future Work (To consider more)
# 
# **To Be Developed (TBD)**  
# * Time-Series plots
#     * Year 2015, 2016, 2017, 2018: Percentage of Standard Exceeded of (current year - last year)] 
# * Histogram
#     * House price correlation with the test performance (bin 10: house_cost <25K, ...., house_cost > 100k)
# * More external factors
#     * Teachers demographics
# * Bokeh or Plotly libraries 
#     * Switch barplots to interactive graphs
#     * Longitude, latitude information to visualize the score distribution on the map (Plotly libraries)
# * Use more of special metrics
#     * Ex) Score_gap = |Standard Exceeded % - Standard Not Met %|
# 
# **Need to be more consider**  
# * Not all data available for subgroups - skewed?
# * Need to eliminate outliers?
#     * Ex) (new schools (e.g., Eastwood elementary), etc.) - top and bottom 5% schools need to be excluded.
# * Further analysis on the High-income whites vs. Low-income whites 
#     * Ex) (df.loc[(df['House_median'] > 130k) & (df['Subgroup Id'] == whites)]
# * More insights? Trends? More hypothesis?
#     * Much more score differnces in mathematics in (high schools groups / minority groups)

#%%
#replace plotly -> bokeh
#cf.set_config_file(offline=False, world_readable=True, theme='ggplot')
#math_top.iplot(kind='bar', barmode='stack', filename='cufflinks/stacked-bar-chart') 

#%% [markdown]
# # Miscellaneous
#%% [markdown]
# **Functions**

#%%
def getScoreLevel(df):
    scoreLevel = ''
    for col in df.columns:
        if col == "County Name" or col == "District Name" or col == "School Name":
            if not df[col].isnull().any().any():
                print(col)
                scoreLevel = re.sub('Name', '', col)
    return scoreLevel

def get_id_vals(df):
    final_id_vars = []   
    scoreLevel =  getScoreLevel(df)    
    if "School" in scoreLevel:
        final_id_vars = final_id_vars_schools
    elif "District" in scoreLevel:
        final_id_vars = final_id_vars_districts
    elif "County" in scoreLevel:
        final_id_vars = final_id_vars_counties
    else:
        print("no matching level")   
    return final_id_vars


#%%
#Create a column "ratio"
#scoreLevel: "School Name", "District Name", "County Name"
def createColumnApply(df, scoreLevel, category, sum_col, new_col):

    #flatten multi index for the results obtained from groupby
    df_sum = df.groupby([scoreLevel, category], as_index=False).sum()
    #create new column "Scores_sum": sum grouped by category 
    #Example: new_col = "Scores_sum", sum_col = "Students with Scores"
    df_sum[new_col] = df_sum[sum_col]

    df_newcol = pd.merge(df, df_sum[[scoreLevel, category, new_col]], 
                                           how='left', on=[scoreLevel, category])

    #create a new column using lambda func.
    #axis = 1 or ‘columns’: apply function to each row
    #apply function should be modified for the usage
    df_newcol["ratio"] = df_newcol.apply(lambda x: x[sum_col]/x[new_col]*100, axis=1)
    #Formatted to second decimal place
    df_newcol = df_newcol.round(2)

#%% [markdown]
# **School Types**  
# We could not find any special results regarding the school type.

#%%
final_data_school_all_ex = pd.melt(final_data_school_all, id_vars=final_id_vars, value_vars=final_value_vars, var_name='Performance Group', value_name='Performance Group Percentage')


def drawBarplotFacetGridEx2(df, _col, perfCols):
    df_local = df.loc[df['Performance Group'].isin(perfCols)]
    
    g = sns.FacetGrid(df_local, col=_col, height=6)
    g.map(sns.barplot, "Type Id", "Performance Group Percentage", 'Performance Group', palette="rocket", ci=None )
    g.add_legend()
    g.set_xticklabels(rotation=90)
    plt.show()

drawBarplotFacetGridEx2(final_data_school_all_ex, "Test Id", ['Percentage Standard Exceeded', 'Percentage Standard Not Met'])

#%% [markdown]
# **Specific schools, district, or counties (CDS)**  
# 
# To obtain the test scores of specific schools, districts, or counties, the **exact school codes** needs to be retrieved from entity tables. When finding the school codes, you should specify a **county**, a **district**, and a **school** names because there may exist several schools with the same names. These are denoted as the **‘CDS’**. 
#%% [markdown]
# For example, if we want the obtain the school code of the `Eastwood Elementary` school in `Irvine` district and `Orange` county, we first select the DataFrame using the names of the **county**, **district**, and **school** conditions all together.
#%% [markdown]
# Please note that if we specify only the school name(s), we could retrieve the several schools with the same names. It is important to include these three codes to avoid the double-counting in any summary calculations.

#%%
final_data = final_data_school_all
final_data.loc[(final_data['School Name'] == 'Eastwood Elementary') & 
             (final_data['District Name'] == 'Irvine Unified') & 
             (final_data['County Name'] == 'Orange'), :].head()

#%% [markdown]
# We can retrieve the DataFrames of the **county** and the **district**. For example, we can obtain the test score of `Irvine Unified` District as follows.

#%%
# Irvine Unified Code
irvine_district = final_data.loc[(final_data['District Name'] == 'Irvine Unified') & 
               (final_data['County Name'] == 'Orange'), :]
irvine_district.head()


#%%
#retreive only "Elementary schools"
irvine_district_elementary = irvine_district[irvine_district["School Name"].str.contains('Elementary')]
irvine_district_elementary.head()


#%%
dfTop_Low_irvine = getDFTop_Low(irvine_district_elementary, "School Name", 5)

drawBarplotCatplot1(dfTop_Low_irvine, "All Students", "English", "School Name", "Percentage Standard Exceeded", "Rank", "Test Id")
drawBarplotCatplot1(dfTop_Low_irvine, "All Students", "English", "School Name", "Students with Scores", "Rank", "Test Id")


drawBarplotCatplot1(dfTop_Low_irvine, "All Students", "Mathematics", "School Name", "Percentage Standard Exceeded", "Rank", "Test Id")
drawBarplotCatplot1(dfTop_Low_irvine, "All Students", "Mathematics", "School Name", "Students with Scores", "Rank", "Test Id")


