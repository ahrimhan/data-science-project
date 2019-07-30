## **[Capstone Project 1: Prediction of Scores for Public Schools in California](https://github.com/ahrimhan/data-science-project/tree/master/project1)**


### Documention
1. [Final Report](https://github.com/ahrimhan/data-science-project/blob/master/project1/reports/capstone1_final_report.pdf)
2. [Presentation](https://github.com/ahrimhan/data-science-project/blob/master/project1/reports/capstone1_presentation.pdf)

### Jupyter Notebooks
1. [Data Wrangling](https://github.com/ahrimhan/data-science-project/blob/master/project1/data_wrangling.ipynb)
2. [Data Visualization](https://github.com/ahrimhan/data-science-project/blob/master/project1/data_visualization.ipynb)
3. [Exploratory Data Analysis](https://github.com/ahrimhan/data-science-project/blob/master/project1/exploratory_data_analysis.ipynb)
4. [Machine Learning](https://github.com/ahrimhan/data-science-project/blob/master/project1/machine_learning.ipynb)


**We have analyzed the California Assessment of Student Performance and Progress (CAASPP) score data (California Department of Education)
and house prices (Zillow research data) to help predict and find the inferior groups of schools that indeed need help.**

In the data visualization and exploratory data analysis, we plotted the various kinds of graphs interactive stacked bars using Plotly library and gained the insights on exceeded scores and inferior scores regarding to gender, ethnicity, english-language fluency, economic status, disability status, and parent educations.
We also performed correlation analysis, univariate selection, and feature importance methods to find the strong indicators affecting lower scores. 

In the modeling, we used the supervised machine learning  algorithms including the regression and classification to build predictive models.
The regression algorithm predicts the percentage of students who do not meet the standard. 
The classification algorithm predicts if the schools "need help" (1) or "do not need help" (0). 
We set the "need help" schools that has more than "80\% of the standard not met" students (312 out of 8,786 schools).
We tried various machine learning techniques to pick the one which performs best.
For regression, out of 5 different models, we obtained the best regression model using the random forest regressor with 10 folds cross validation
with the accuracy of RMSE 10.77, MAE 7.69, and R<sup>2</sup> 0.68.
For classification, we tried to solve the class imbalanced problems using the Stratified K-fold cross validation and the weighted evaluation metrics to reflect the mass of the classes. In addition, we scaled the training data and significantly improved the accuracy of the K-Nearest Neighbor algorithm.
As a result, we obtained the best classification model using the random forest classifier based on grid search cross validation with the accuracy 0.97 and AUC 0.98.

Based on these results, we identified the top and bottom schools and found the important features determining those schools.
We recommended some strategies
that effectively increase the achievements for scores. 
