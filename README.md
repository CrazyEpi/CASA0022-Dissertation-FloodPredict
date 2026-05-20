# CASA0022-Dissertation-FloodPredict

## Project Overview
This project aims to build a 24-hour advance flood warning system for the House Mill, a historical building in London, with Internet of Things (IoT) data and deep learning models. As the largest surviving tidal mill in the UK, House Mill is frequently threatened by compound flooding, making the projection of the house mill a critial issue.

By utilizing ultrasonic water level sensors deployed on-site, combined with meteorological and tidal API data provided by the Environment Agency (EA), I have built an automated data pipeline. The core objective of this project is to compare and optimize the predictive performance of different time series forecasting models (such as LSTM, Informer, and Autoformer) in extreme flood events, thereby assisting local volunteers in making flood protection decisions.

## Current Progress
Currently, I have successfully completed data collection and cleaning, and have preliminarily finished the deployment and iteration of the LSTM model:

1. Baseline Deployment: I first reproduced the classic LSTM model proposed by Kratzert et al. (2018) in the paper Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks as our baseline.

2. Baseline Performance Analysis: In initial tests, the performance of the model identical to Kratzert's paper was not ideal. I believe the fundamental reason is that the nature of the task is different: Kratzert's model is primarily used for predicting continuous rainfall-runoff, whereas the flooding at House Mill is an extreme event.

3. Improvement: To address the above issues, I customized a LSTM model that to make sure the floods are focused by the model, and achieved a significant progress compared to the baseline. TODO: add architechture information

## Next Steps: 

Add rainfall data and begin developing Informer and Autoformer models
