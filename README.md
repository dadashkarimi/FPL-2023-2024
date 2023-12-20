# FPL Player Ranking using Machine Learning

This repository contains code for ranking Premier League players based on their performance in the 2023-24 season and previous seasons using various machine learning methods. We compare the effectiveness of four different models: a simple ensemble-based decision tree, Lasso regression, a convolutional neural network (CNN), and a long short-term memory (LSTM) network.

## Overview
The goal of this project is to predict player performance and rank them using machine learning techniques. We demonstrate how different models perform in ranking players based on their historical performance and current statistics.

###  Methods Used
Models:
1. Decision Tree Ensemble: Utilized an ensemble method to predict player rankings.
2. Lasso Regression: Applied L1 regularization for player ranking prediction.
3. Convolutional Neural Network (CNN): Implemented a CNN for player ranking based on historical data.
4. Long Short-Term Memory (LSTM) Network: Employed an LSTM to predict player rankings, capturing sequential dependencies in player performance.

## Results
Performance Comparison:
Conducted various simulations and evaluations over past game weeks.
Demonstrated that CNN and LSTM models consistently outperform decision tree and Lasso regression methods.
Evaluated the models' performance metrics such as Mean Squared Error (MSE), Accuracy, and R-Squared across different experiments.
Recommendations
Based on the models' predictions and historical analysis, we provide a list of recommended transfer candidates for the upcoming game week. These suggestions are derived from the analysis of player rankings and performance predictions.

|      | LGBM      | Lasso     | CNN       |
|------|-----------|-----------|-----------|
| 15   | 4.47      | 14.38     | 2.12      |
| 16   | 3.53      | 12.69     | 1.95      |
| 17   | 4.42      | 6.58      | 2.12      |
| 18   | 2.63      | 7.04      | 1.49      |

## Usage
Requirements:
Python 3.x
Libraries: Torch, NumPy, Pandas, Scikit-learn
Instructions:
Clone this repository: git clone [https://github.com/dadashkarimi/FPL-2023-2024.git](https://github.com/dadashkarimi/FPL-2023-2024.git)
Install required dependencies: pip install -r requirements.txt
Run the notebooks or scripts in the src directory to train models and predict player rankings.
Explore model comparisons and transfer recommendations in the respective notebooks or files.

## Directory Structure
datasets/: Contains datasets of players, teams, and their performances in previous seasons
predicted_dataset/: Contains predicted datasets for all methods
