# NFL Combine 40-Yard Dash Prediction Model

This project uses historical NFL Combine data to build a predictive model for 40-yard dash times using random forest regression.
  - Also additional analyses included

## Project Overview
- Goal: Predict 40-yard dash times for NFL prospects based on physical traits and combine drill performance.
- Model: Random Forest Regression
- Performance:
  - R² = 0.855
  - RMSE ≈ 0.11 seconds
  - MAE ≈ 0.088 seconds
- Tools Used: R, dplyr, randomForest, Tableau

## Files Included
- `combine.R`: Full R script including data cleaning, feature engineering, modeling, and visualizations
- `combine_data.xlsx`: Cleaned and organized dataset used to train and evaluate the model (2000–2025 NFL Combine data)
  - The original data was scraped and cleaned from publicly available NFL data on https://www.pro-football-reference.com/
- `NFL Combine Project.pdf`: Final written report summarizing methodology, findings, and model insights
- `NFL Combine Project - Charts and Visualizations.pdf`: Collection of residual plots, prediction accuracy charts, and other visuals
- `NFL Combine Slides - MS Practicum.pdf`: Slide deck version of the project, created for formal presentation  
- `README.md`: Summary and documentation for the project

## Key Features Used
- Physical traits: height, weight, position
- Combine drills: bench press, vertical jump, shuttle run, broad jump
- Metadata: year, position group

## Model Insights
- Variable importance analysis identified weight, vertical jump, and shuttle time as the most predictive features.
- The model performs well across most positions, with analysis revealing performance outliers and group-based trends.
- Residuals are generally low, and visual diagnostics indicate strong model fit and reliable generalization.

## Contact
Created by Spencer Slote  
spencerslote8302@gmail.com  
[LinkedIn](https://www.linkedin.com/in/spencer-slote-576a3729a/)
