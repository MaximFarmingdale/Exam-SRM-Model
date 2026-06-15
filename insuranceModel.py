# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:23:07 2026

@author: Maxim
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn as skl
import polars as pl 
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

#removed max torque and max power as they are hard to interpert 
data = pl.read_csv("InsuranceClaimsData.csv").drop(["max_torque","max_power", "policy_id"])
data = data.to_dummies(["fuel_type", "engine_type", "transmission_type", "steering_type", "model"], drop_first=True)
data = data.with_columns(pl.lit(1.0).alias("constant"))
yes_no_columns = ["is_esc", "is_adjustable_steering", "is_tpms", "is_parking_sensors"]
data = data.with_columns(
    pl.col(yes_no_columns)
    .str.to_lowercase()
    .replace_strict({
        "yes": 1, 
        "true": 1, 
        "no": 0,
        "false": 0
    }, default=None)
    .cast(pl.Int8))
modelNames = ["model_M1", "model_M2", "model_M3", "model_M5", "model_M6", "model_M7", "model_M8", "model_M9" , "model_M10", "model_M11"]
data = data.select(["vehicle_age", "subscription_length", "customer_age", "region_density", "claim_status"] +modelNames)
vifData = data.drop(["claim_status"]).select(pl.selectors.numeric())
vifColumns = vifData.columns
vifData = vifData.to_numpy()
vif = [variance_inflation_factor(vifData, i) for i in range(vifData.shape[1])]
vifDf = pl.DataFrame({
    "Variable": vifColumns,
    "VIF": vif
    }).sort("VIF", descending=True)
print("multicollinearity test\n")
print(vifDf)
#customer_age has a high vif but still doesnt pass the threshold of 10
#LASSO and other penalizing models should counteract this
seed = 1234
explanatory = data.drop("claim_status").to_numpy()
response = data.select("claim_status").to_numpy()
response_ravel = response.ravel()
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state= seed)
# removed the scalers for the excel part but never repeat this otherwise, it sucks 
pipelineLasso = Pipeline([
    ('classifier', LogisticRegression(
        penalty = "elasticnet",
        l1_ratio = 1.0, 
        solver='saga', 
        C = .1, 
        max_iter = 10000,
        random_state=seed
    ))])
pipelineRidge = Pipeline([
    ('classifier', LogisticRegression(
        penalty = "elasticnet",
        l1_ratio = 0,
        solver='saga', 
        C = .1, 
        max_iter = 10000,
        random_state=seed
    ))])
pipelineWeighted = Pipeline([
    ("classifier", LogisticRegression(
        penalty=None,
        class_weight="balanced", 
        solver = "lbfgs",
        random_state=seed
    ))])
randomForest = RandomForestClassifier(n_estimators=100, max_depth= 10, random_state= seed)
boosting = LGBMClassifier(n_estimators=100, max_depth= 3, random_state=seed)
namesOfRegModels = ["Lasso", "Ridge", "Weighted Least Squares", "Random Forrest", "Boosting"]

cv_scores = {
    "Models":namesOfRegModels,
    "Score": [
        cross_val_score(pipelineLasso, explanatory, response_ravel,cv = cv, scoring="roc_auc").mean(),
        cross_val_score(pipelineRidge, explanatory, response_ravel,cv = cv, scoring="roc_auc").mean(),
        cross_val_score(pipelineWeighted, explanatory, response_ravel,cv = cv, scoring="roc_auc").mean(),
        cross_val_score(randomForest, explanatory, response_ravel,cv = cv, scoring="roc_auc").mean(),
        cross_val_score(boosting, explanatory, response_ravel,cv = cv, scoring="roc_auc").mean()
    ]
    }
#boosting got the highest cv score and lasso got the highest out of the tree=based models
#we can assume that the relationship between variables has nonlinear aspects that both of the tree based models picked up on
pipelineLasso.fit(explanatory, response_ravel)
pipelineRidge.fit(explanatory, response_ravel)
pipelineWeighted.fit(explanatory, response_ravel)

lassoCoefs = pipelineLasso.named_steps["classifier"].coef_[0]
ridgeCoefs =  pipelineRidge.named_steps["classifier"].coef_[0]
weightedCoefs =  pipelineWeighted.named_steps["classifier"].coef_[0]

lassoIntercept = pipelineLasso.named_steps["classifier"].intercept_[0]
ridgeIntercept = pipelineRidge.named_steps["classifier"].intercept_[0]
weightedIntercept = pipelineWeighted.named_steps["classifier"].intercept_[0]
coefs = {
    "Names": ["Lasso", "Ridge", "Weighted"],
    "Coef": [lassoCoefs, ridgeCoefs, weightedCoefs]
}
print(f"{lassoIntercept} {ridgeIntercept} {weightedIntercept}")
