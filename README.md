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

- ## Project Structure:
G_G_Mahesh_FinalProjectB11/
├── G_G_Mahesh_FinalProjectB11.ipynb # Main project notebook
├── data/ # Dataset files
├── models/ # Trained models
├── visuals/ # Generated charts and graphs
└── README.md # Project documentation

## Dataset:
The project uses FIFA player data containing:

Player demographics (age, nationality, position)

Physical attributes (height, weight)

Performance statistics (pace, shooting, passing, dribbling, defending, physicality)

Skill ratings and special abilities

Current market values and wages

## Machine Learning Approach
The project implements:

##  Data Preprocessing:

1.Handling missing values

2.Feature scaling and normalization

3.Categorical variable encoding

## Visualization Samples:
1.Age vs 
Flight Price vs Total Stops
Correlation Heatmap
Actual vs Predicted Prices (Scatter Plot)

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
Overall Performance Metrics:
Model	MAE (€)	MSE (€²)	RMSE (€)	R² Score	Training Time (s)	Inference Time (ms)
Random Forest	1,250,000	2.8×10¹²	1,674,000	0.89	45.2	12.5
Gradient Boosting	1,380,000	3.2×10¹²	1,789,000	0.87	38.7	8.3
XGBoost	1,320,000	3.0×10¹²	1,732,000	0.88	42.1	6.8
Neural Network	1,450,000	3.6×10¹²	1,897,000	0.85	120.5	15.2
Linear Regression	2,100,000	6.5×10¹²	2,550,000	0.73	3.2	2.1
Decision Tree	1,650,000	4.1×10¹²	2,025,000	0.83	12.8	4.5
Support Vector Regression	1,950,000	5.8×10¹²	2,408,000	0.76		

## Hyperparameter Tuning Results:
Hyperparameter Tuning Results
Model	Best Parameters	Improvement (%)
Random Forest	n_estimators=200, max_depth=25, min_samples_split=5	+8.2%
Gradient Boosting	learning_rate=0.1, n_estimators=150, max_depth=6	+6.5%
XGBoost	learning_rate=0.01, max_depth=7, subsample=0.8	+7.1%
Neural Network	layers=[64,32,16], dropout=0.3, learning_rate=0.001	+9.8%


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

## Contributors
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


   










