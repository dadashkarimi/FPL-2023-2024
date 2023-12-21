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
| 2  | 4.2763607287| 14.41133689| 2.11351848| 2.11440730|
| 3  | 3.3828815951| 13.58604145| 1.92744124| 1.92659879|
| 4  | 4.8160802119| 4.907651070| 4.41780757| 2.25008749|
| 5  | 3.3846687088| 3.546627281| 3.72422599| 1.93738008|
| 6  | 4.3555052590| 4.480956995| 4.27773762| 2.17485380|
| 7  | 4.2732424385| 4.139855709| 4.11673570| 2.04702878|
| 8  | 3.6951737614| 3.710253780| 3.82981634| 1.92654407|
| 9  | 3.7319135866| 3.875399451| 3.90936350| 2.00428367|
| 10 | 3.9380178953| 3.924139776| 3.96490765| 1.97034812|
| 11 | 4.1276260457| 4.084303992| 4.14363241| 2.05053163|
| 12 | 3.7137212790| 3.750264118| 3.91302800| 1.94224727|
| 13 | 7.7606873301| 8.119572649| 5.89781988| 3.91735685|
| 14 | 8.3032367393| 8.236854604| 6.14845371| 6.19199061|
| 15 | 13.3972699467| 17.98069733| 8.54950118| 6.36534309|
| 16 | 7.0587534334| 11.36677232| 3.86536741| 3.87041473|
| 17 | 8.8340433814| 9.991549395| 2.05676436| 2.07859159|


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
