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
We ran multiple rounds of simulations and tests on previous game weeks. The results? Well, the CNN clearly stole the show! 

|    |     lgbm     |    lasso    |    cnn     |    lstm    |
|----|-------------|------------|-----------|-----------|
| 16 | 3.5293767167| 3.7889241068| 1.92181015| 1.93569374|
| 17 | 8.8340433814| 5.6575293541| 2.05676436| 2.07859159|
| 18 | 2.6269001574|13.312789917 | 1.46523809| 1.59688509|

## Usage
Requirements:
Python 3.x
Libraries: Torch, NumPy, Pandas, Scikit-learn
Instructions:
Clone this repository: git clone [https://github.com/dadashkarimi/FPL-2023-2024.git](https://github.com/dadashkarimi/FPL-2023-2024.git)
Install required dependencies: ``
pip install -r requirements.txt
``
Run the notebooks or scripts in the src directory to train models and predict player rankings.
Explore model comparisons and transfer recommendations in the respective notebooks or files.

## Directory Structure
datasets/: Contains datasets of players, teams, and their performances in previous seasons
predicted_dataset/: Contains predicted datasets for all methods

# Acknowledgements
We truely appreciate the code and data that are shared by [https://github.com/saheedniyi02/fpl-ai.git](https://github.com/saheedniyi02/fpl-ai.git) to download and curate player stats from FPL. 
