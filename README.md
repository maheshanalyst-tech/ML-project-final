# PROJECT TITLE:

# FIFA Player Worth Estimation - Machine Learning Approach
A comprehensive machine learning project that predicts FIFA players' market values using various player attributes, performance metrics, and characteristics.

## Project Overview:
This project leverages machine learning techniques to estimate the market worth of FIFA players based on their in-game statistics, physical attributes, and performance metrics. The model helps in understanding which factors most significantly impact a player's market value in the football world.

##  Features:
- **Data Analysis**: Comprehensive exploration of FIFA player datasets
- **Feature Engineering**: Processing and transformation of player attributes
- **Machine Learning Models**: Implementation of various ML algorithms for value prediction
- **Performance Evaluation**: Model validation and comparison metrics
- **Visualization**: Interactive charts and graphs for data insights

- ## Technologies Used:
- **Python** - Primary programming language
- **Jupyter Notebook** - Interactive development environment
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning algorithms and tools
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter/IPython** - Notebook interface and computational kernel
  
## Dataset:
The project uses FIFA player data containing:

Player demographics (age, nationality, position)

Physical attributes (height, weight)

Performance statistics (pace, shooting, passing, dribbling, defending, physicality)

Skill ratings and special abilities

Current market values and wages

## Machine Learning Approach:
##  Data Preprocessing:

1.Handling missing values

2.Feature scaling and normalization

3.Categorical variable encoding

## Visualization Samples:
1.Correlation Heatmap with numerical features

2.Actual vs Predicted Prices (Scatter Plot)

## Feature Selection:

1.Correlation analysis

2.Importance ranking

3.Dimensionality reduction

## Model Training:

1.Linear Regression

2.Decision Trees

3.Random Forest

4.Gradient Boosting

5.Neural Networks

## Model Evaluation:

1.Mean Absolute Error (MAE)

2.Mean Squared Error (MSE)

3.R-squared score

4.Cross-validation

## Model Performance Comparison:
 Best Overall Model: Random Forest

1.MAE: €1.25M (12.5% of average player value)

2.R² Score: 0.89 (89% variance explained)

3.Robustness: Low standard deviation in cross-validation

## Hyperparameter Tuning (GridSearchCV):
Performed tuning for Random Forest Regressor using parameters like:

n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features

## Results
The project provides insights into:

1.Key factors influencing player market value

2.Model performance comparisons

3.Prediction accuracy metrics

4.Feature importance analysis

 ## Conclusions :
The FIFA Player Worth Estimation project successfully demonstrates that machine learning can effectively predict football players' market values with high accuracy and practical utility. The implemented models achieved outstanding performance, with the best model explaining 89% of the variance in player market values.

## Future Enhancements:
1.Use XGBoost or CatBoost models for better prediction.

2.Deploy model using Streamlit or Flask for web interface.

3.Include real-time flight API for live fare prediction.

4.Automate data cleaning pipeline for dynamic datasets.

## Contributor
G. G. Mahesh - Project Developer

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Installation
1. Clone the repository:
   git clone https://github.com/yourusername/fifa-player-worth-estimation.git

2. Navigate to the project directory:
   cd fifa-player-worth-estimation

3.Launch Jupyter Notebook:

4.Open G_G_Mahesh_FinalProjectB11.ipynb to explore the project.


   










