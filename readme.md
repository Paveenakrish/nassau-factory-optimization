# Nassau Factory Optimization

## Project Overview

Nassau Factory Optimization is a machine-learning-based decision-support system designed to analyze shipping lead times and evaluate potential factory reallocations.

The project combines exploratory data analysis, regression modeling, factory performance analysis, and ML-based what-if simulation in an interactive Streamlit dashboard.

## Objectives

- Analyze shipping lead-time patterns
- Compare shipping performance across products, factories, regions, and shipping modes
- Train machine-learning models to predict shipping lead time
- Identify products with above-average lead times
- Simulate alternative factory assignments
- Evaluate predicted lead-time improvements
- Assess scenario confidence, risk, and profitability
- Provide factory reallocation recommendations through an interactive dashboard

## Dataset

The dataset contains 10,194 records and 18 original columns covering:

- Order information
- Customer geography
- Product information
- Factory/division information
- Shipping mode
- Sales
- Units
- Gross Profit
- Cost
- Order Date
- Ship Date

## Machine Learning Models

Three regression models were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

### Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 181.4749 | 182.5624 | 0.5287 |
| Random Forest | 151.5239 | 170.1499 | 0.5906 |
| Gradient Boosting | 155.8964 | 168.4407 | 0.5988 |

Gradient Boosting was selected as the candidate final model because it achieved the lowest RMSE and highest R² among the evaluated models.

## Factory Optimization

The optimization workflow includes:

1. Product-level factory performance analysis
2. Reallocation candidate identification
3. ML-based factory what-if simulation
4. Scenario confidence and risk scoring
5. Profitability impact analysis
6. Final recommendation scoring

The final ML scenario analysis identified two products with low-risk positive predicted improvements:

- Wonka Bar - Triple Dazzle Caramel
- Wonka Bar - Nutty Crunch Surprise

These recommendations are intended as decision-support scenarios and should be operationally validated before implementation.

## Dashboard

The Streamlit application provides:

- KPI overview
- Factory performance comparison
- Product shipping performance
- Region and shipping-mode filtering
- Factory what-if simulation
- ML-based factory recommendations
- Data-quality warning and assumptions

## Project Structure

```text
nassau factory optimization/
│
├── app.py
├── readme.md
├── requirements.txt
│
├── data/
│   └── dataset.csv
│
├── notebook/
│   ├── nassau_analysis.ipynb
│   └── nassau_streamlit_backend.ipynb
│
└── source code/
    ├── init.py
    ├── model.py
    ├── optimization.py
    └── preprocessing.py