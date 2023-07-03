# disasterprep
This is an attempt at using machine learning and capabilities of AI(Artificial Learning) at predicting disasters before they occur. Even though Natural disasters have a lot of factors impacting them and at a really large scale , they are often unpredictable but a short term notice before they occur can be given to the public under the effect so as to save lives wth the help of machine learning model and time series analysis.Machine learning models can recognise patterns that often humans are uncapable of and can process a large amount of data at a time.

 # This is DisasterPrep : A predicting and Analysis Tool for Flood, Rainfall and Medical Emergencies

 # Flood Prediction
  this is done using LD(for past analysis) and FbProphet(for time series analysis) and thus predicting for several rivers on the basis of their runoff values if the low lying areas can face flood or not 

 # Rainfall Analysis
 this is done using RNN(Recurrent Neural Network) to analyse and train the model to predict each year's rainfall and display the accuracy and mean squared error on the side pane.

 # Medical Emergency Prediction
 this is a really hopeful part , here the data from local hospital is collected on statistics of disease , (only available data was rio di janerio's dataset on covid19) and then ARIMA model is used to predict the time series analysis of the disease. In this way if a particular locality is facing serious numbers of cases, the citizens can be notified about it and henceforth advised to take prevention measures.

![flowchart](https://github.com/anjali-113/disasterprep/assets/127101288/3e74ccc1-ae97-42b1-8485-f19eb6cdc79b)


 # How to launch this
 install all the reuired dependencies 
 -flask
	-wtforms
	-fbprophet
	-sklearn
	-Keras
	-Tensorflow
 and then launch main.py, the link generated is the localhost link and the website will be hosted.
 it is prefered to launch in a virtual environment so that the libraies and modules versions do not interfere.
	-Imblearn
