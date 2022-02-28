#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



Author	: Nirbhai Singh
E-Mail	: chittamor@gmail.com


*********************************

Run below commands to verify you are in correct environment and branch:
    !conda info
    !git branch --show-current

*********************************


"""

# for handling address of directories where data is stored
from pathlib import Path

# for reading data
import pandas as pd

# for mathematical functions
import numpy as np

# for plotting graphs
import plotly.express as px
import plotly.graph_objects as go

# below three lines helps choose where to render the plotly plots
# it needs plotly dependency kaleido installed for in-IDE rendering as svg
import plotly.io as pio
pio.renderers.default = 'svg'
#pio.renderers.default = 'browser'

# for machine learning models
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# for machine learning model evaluation metrics
from sklearn.metrics import r2_score, mean_squared_error
# custom package by authors of practical statistics for data scientists
from dmba import AIC_score
from dmba import stepwise_selection

# for type annotations and hinting
from typing import List



#-------------------------------------------------------------------------------



# setting the path to data directory
DATA = Path().resolve() / 'data'

lung_df = pd.read_csv(DATA/'LungDisease.csv')

fig = px.scatter(
                lung_df,
                x = 'Exposure',
                y = 'PEFR'
                )
fig.show()



#-------------------------------------------------------------------------------



# using simple linear regression to estimate PEFR

feature_vector = ['Exposure']
target = 'PEFR'

model = LinearRegression()
model.fit(lung_df[feature_vector], lung_df[target])

print(f"Intercept: {model.intercept_:.3f}")
print(f"Coefficient: {model.coef_[0]:.3f}")
print("-----------------------------------------------------------------------")

fitted = model.predict(lung_df[feature_vector])
residuals = lung_df[target] - fitted

# one way of plotting the regression line is with "trendline" property 
# in plottly express scatter plot
fig = px.scatter(
                lung_df,
                x = 'Exposure',
                y = 'PEFR',
                trendline='ols',
                trendline_color_override='red'
                )
fig.show()


# another way of plotting the same graph:
#  - without using trendline for regression line
#    instead using predicted values from a fitted model for plotting regression line
#  - without using plotly.express
#    instead directly using plotly.graph_objects on an empty figure

# plotly figure setup
fig = go.Figure()
fig.add_trace(
                go.Scatter(
                    name = "PEFR vs Exposure",
                    x = lung_df['Exposure'],
                    y = lung_df['PEFR'],
                    mode = 'markers'
                    )
                )
fig.add_trace(
                go.Scatter(
                    name = "line of best fit",
                    x = lung_df['Exposure'],
                    y = fitted,
                    mode = 'lines'
                    )
                )

# plotly figure layout
fig.update_layout(xaxis_title = 'PEFR', yaxis_title = 'Exposure')
fig.update_layout(legend = dict(
                                orientation = 'h',
                                yanchor = 'bottom',
                                y = 1.02,
                                xanchor = 'right',
                                x = 1))

fig.show()



#-------------------------------------------------------------------------------



# plotting residuals from the regression line to illustrate the fit of model

# 1st way - using trendline and plotly.express.scatter
fig = px.scatter(
                 lung_df,
                 x = "Exposure",
                 y = "PEFR",
                 trendline="ols",
                 trendline_color_override="red"
                 )

# retrieve x-values from one of the series
xVals = fig.data[0]['x']

# container for prediction errors
errors = {}

# organize data for errors in a dict
for d in fig.data:
    errors[d['mode']]=d['y']

# container for shapes
shapes = []

# make a line shape for each error == distance between each marker and line points
for i, x in enumerate(xVals):
    shapes.append(go.layout.Shape(type="line",
                                    x0=x,
                                    y0=errors['markers'][i],
                                    x1=x,
                                    y1=errors['lines'][i],
                                    line=dict(
                                        color = 'black',
                                        width=1,
                                    dash = 'dot'),
                                    opacity=0.5,
                                    layer="above")
                 )

# include shapes in layout
fig.update_layout(shapes=shapes)

fig.show()

# 2nd way, using plotly.graph_objects on an empty figure and a custom model

# plotly figure setup
fig=go.Figure()
fig.add_trace(
                go.Scatter(
                    name = "PEFR vs Exposure",
                    x = lung_df['Exposure'],
                    y = lung_df['PEFR'],
                    mode = 'markers'
                    )
                )
fig.add_trace(
                go.Scatter(
                    name = "line of best fit",
                    x = lung_df['Exposure'],
                    y = fitted,
                    mode = 'lines'
                    )
                )

# plotly figure layout
fig.update_layout(xaxis_title = 'PEFR', yaxis_title = 'Exposure')

# retrieve x-values from one of the series
xVals = fig.data[0]['x']

# container for prediction errors
errors = {}

# organize data for errors in a dict
for d in fig.data:
    errors[d['mode']]=d['y']

# container for shapes
shapes = []

# make a line shape for each error == distance between each marker and line points
for i, x in enumerate(xVals):
    shapes.append(go.layout.Shape(type="line",
                                    x0=x,
                                    y0=errors['markers'][i],
                                    x1=x,
                                    y1=errors['lines'][i],
                                    line=dict(
                                        color = 'black',
                                        width=1,
                                        dash = 'dot'),
                                    opacity=0.5,
                                    layer="above")
                 )

# include shapes in layout
fig.update_layout(shapes=shapes)

fig.update_layout(legend = dict(
                                orientation = 'h',
                                yanchor = 'bottom',
                                y = 1.02,
                                xanchor = 'right',
                                x = 1))

fig.show()



#-------------------------------------------------------------------------------



# using multiple linear regression to estimate Sale_value of houses

HOUSE_CSV = DATA / 'house_sales.csv'
house_df = pd.read_csv(HOUSE_CSV, sep = '\t')

subset = ['AdjSalePrice', 'SqFtTotLiving',
          'SqFtLot', 'Bathrooms',
          'Bedrooms', 'BldgGrade']
print(house_df[subset].head())
print("-----------------------------------------------------------------------")

feature_vector = ['SqFtTotLiving',
          'SqFtLot', 'Bathrooms',
          'Bedrooms', 'BldgGrade']

target = 'AdjSalePrice'

house_lm = LinearRegression()
house_lm.fit(house_df[feature_vector], house_df[target])

print(f"Intercept: {house_lm.intercept_:.3f}")
print("Coefficients:")
for name, coef in zip(feature_vector, house_lm.coef_):
    print(f"    {name}: {coef}")
print("-----------------------------------------------------------------------")

# assessing the model

fitted = house_lm.predict(house_df[feature_vector])
RMSE = np.sqrt(mean_squared_error(y_true = house_df[target], y_pred = fitted))
r2 = r2_score(y_true = house_df[target], y_pred = fitted)
print(f"RMSE: {RMSE:.0f}")
print(f"r2: {r2:.4f}")
print("-----------------------------------------------------------------------")

# statsmodels package gives more detailed analysis of the regression model

model = sm.OLS(house_df[target], house_df[feature_vector].assign(const=1))
results = model.fit()
print(results.summary())
print("-----------------------------------------------------------------------")



#-------------------------------------------------------------------------------



# model selection and stepwise regression

feature_vector = [
                    'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 
                    'Bedrooms', 'BldgGrade', 'PropertyType',
                    'NbrLivingUnits','SqFtFinBasement',
                    'YrBuilt', 'YrRenovated', 'NewConstruction'
                  ]

# we need to convert categorical and boolean variables into numbers
# in order to use statsmodels linear regression

# categorical variable handling
XX = pd.get_dummies(house_df[feature_vector], drop_first=True)

# boolean variable handling
XX['NewConstruction'] = [1 if nc else 0 for nc in XX['NewConstruction']]

model = sm.OLS(house_df[target], XX.assign(const=1))
results = model.fit()
print(results.summary())
print("-----------------------------------------------------------------------")


# adding more features always reduces RMSE and increases R-squared for the training data
# Hence, these are not appropriate to help guide the model choice
fitted = results.fittedvalues
RMSE = np.sqrt(mean_squared_error(y_true = house_df[target], y_pred = fitted))
r2 = r2_score(y_true = house_df[target], y_pred = fitted)
print(f"RMSE: {RMSE:.0f}")
print(f"r2: {r2:.4f}")
# check and compare these values in console with the previous model's values
print("-----------------------------------------------------------------------")

YY = house_df[target]

#print(type(model))

def train_model(features: List[str]) -> sm.OLS:
    """returns a fitted model for a given set of features"""
    if len(features) == 0:
        return None
    model = LinearRegression()
    model.fit(XX[features], YY)
    return model

def score_model(model: sm.OLS, features: List[str]) -> float:
    """returns AIC_score for a given model and a set of features"""
    if len(features) == 0:
        return AIC_score(y_true = YY,
                         y_pred = [YY.mean()] * len(YY),
                         model = model,
                         df =1
                         )
    return AIC_score(y_true = YY,
                     y_pred = model.predict(XX[features]),
                     model = model
                     )

best_model, best_features = stepwise_selection(XX.columns,
                                               train_model,
                                               score_model,
                                               verbose=True
                                               )

print(f"Intercept: {best_model.intercept_:.3f}")
print("Coefficients:")
for name, coef in zip(best_features, best_model.coef_):
    print(f"   {name}: {coef}")
print("-----------------------------------------------------------------------")

"""
    Penalized regression is similar in spirit to AIC. 
    Instead of explicitly searching through a discrete set of models, 
    the model-fitting equation incorporates a constraint that penalizes 
    the model for too many variables (parameters). 
    Rather than eliminating predictor variables entirely—as with 
    stepwise, forward, and backward selection - penalized regression applies 
    the penalty by reducing coefficients, in some cases to near zero. 
    Common penalized regression methods are ridge regression and lasso regression.
"""

"""
    Stepwise regression and all subset regression are in-sample methods to 
    assess and tune models. This means the model selection is possibly subject 
    to overfitting (fitting the noise in the data) and may not perform 
    as well when applied to new data. One common approach to avoid this is 
    to use cross-validation to validate the models. 
    
    In linear regression, overfitting is typically not a major issue, 
    due to the simple (linear) global structure imposed on the data. 
    For more sophisticated types of models, particularly iterative procedures 
    that respond to local data structure, cross-validation is a very important tool
"""



#-------------------------------------------------------------------------------



# weighted regression

house_df['Year'] = [
                    int(date.split('-')[0])
                    for date in house_df['DocumentDate']
                    ]
house_df['Weight'] = house_df['Year'] - 2005

feature_vector = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 
                  'Bedrooms', 'BldgGrade']
target = 'AdjSalePrice'

house_lm_wt = LinearRegression()
house_lm_wt.fit(house_df[feature_vector], house_df[target], sample_weight=house_df.Weight)

# compare the coefficients in linear and weighted regression
print(pd.DataFrame({
    'predictor': feature_vector,
    'house_lm': house_lm.coef_,
    'house_wt': house_lm_wt.coef_,
}).append({
    'predictor': 'intercept', 
    'house_lm': house_lm.intercept_,
    'house_wt': house_lm_wt.intercept_,
}, ignore_index=True))
print("-----------------------------------------------------------------------")

# compare the residuals (errors) in linear and weighted regression model predictions
residuals = pd.DataFrame({
    'abs_residual_lm': np.abs(house_lm.predict(house_df[feature_vector]) - house_df[target]),
    'abs_residual_wt': np.abs(house_lm_wt.predict(house_df[feature_vector]) - house_df[target]),
    'Year': house_df['Year'],
})

print(pd.DataFrame(([year, np.mean(group['abs_residual_lm']), np.mean(group['abs_residual_wt'])] 
              for year, group in residuals.groupby('Year')),
             columns=['Year', 'mean abs_residual_lm', 'mean abs_residual_wt']))
print("-----------------------------------------------------------------------")



#-------------------------------------------------------------------------------



# factor variables in regression

feature_vector = ['SqFtTotLiving', 'SqFtLot',
                  'Bathrooms', 'Bedrooms',
                  'BldgGrade', 'PropertyType']

XX = pd.get_dummies(house_df[feature_vector], drop_first=True)

house_lm_factor = LinearRegression()
house_lm_factor.fit(XX, house_df[target])

print(f"Intercept: {house_lm_factor.intercept_:.3f}")
print("Coefficients:")
for name, coef in zip(XX.columns, house_lm_factor.coef_):
    print(f"   {name}: {coef}")
print("-----------------------------------------------------------------------")


# factor variables with many levels
print(pd.DataFrame(house_df['ZipCode'].value_counts()).transpose())
print("-----------------------------------------------------------------------")

house_df = pd.read_csv(HOUSE_CSV, sep='\t')

feature_vector = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 
              'Bedrooms', 'BldgGrade']
target = 'AdjSalePrice'

house_lm = LinearRegression()
house_lm.fit(house_df[feature_vector], house_df[target])


zip_groups = pd.DataFrame([
    *pd.DataFrame({
        'ZipCode': house_df['ZipCode'],
        'residual' : house_df[target] - house_lm.predict(house_df[feature_vector]),
    })
    .groupby(['ZipCode'])
    .apply(lambda x: {
        'ZipCode': x.iloc[0,0],
        'count': len(x),
        'median_residual': x.residual.median()
    })
]).sort_values('median_residual')

zip_groups['cum_count'] = np.cumsum(zip_groups['count'])
zip_groups['ZipGroup'] = pd.qcut(zip_groups['cum_count'], 5, labels=False, retbins=False)
print(zip_groups.head())
print("-----------------------------------------------------------------------")

to_join = zip_groups[['ZipCode', 'ZipGroup']].set_index('ZipCode')
house_df = house_df.join(to_join, on='ZipCode')
house_df['ZipGroup'] = house_df['ZipGroup'].astype('category')



#-------------------------------------------------------------------------------



# correlated predictors

feature_vector = ['Bedrooms', 'BldgGrade', 'PropertyType', 'YrBuilt']
target = 'AdjSalePrice'

XX = pd.get_dummies(house_df[feature_vector], drop_first=True)

reduced_lm = LinearRegression()
reduced_lm.fit(XX, house_df[target])

print(f"Intercept: {reduced_lm.intercept_:.3f}")
print("Coefficients:")
for name, coef in zip(XX.columns, reduced_lm.coef_):
    print(f"   {name}: {coef}")
print("-----------------------------------------------------------------------")


# confounding variables

feature_vector = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
              'BldgGrade', 'PropertyType', 'ZipGroup']
target = 'AdjSalePrice'

XX = pd.get_dummies(house_df[feature_vector], drop_first=True)

confounding_lm = LinearRegression()
confounding_lm.fit(XX, house_df[target])

print(f"Intercept: {confounding_lm.intercept_:.3f}")
print("Coefficients:")
for name, coef in zip(XX.columns, confounding_lm.coef_):
    print(f"   {name}: {coef:.3f}")
print("-----------------------------------------------------------------------")

























































































