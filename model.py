import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer,MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
import chart_studio.plotly.plotly as py
import plotly
import joblib

import warnings
warnings.filterwarnings("ignore")
#get_ipython().run_line_magic('matplotlib', 'inline')
#fd-future data set
#validating-0 or 1 (0-tetsing ,1= future prediction)
def flood_classifier(filename,fd,validating=0):

	data1=pd.read_excel('data/'+filename+'.xlsx')

	# In[4]:
	data1.shape
	# In[5]:

	#Fillng null entries with mean of their respective columns
	for i in range(1,len(data1.columns)):
	    data1[data1.columns[i]] = data1[data1.columns[i]].fillna(data1[data1.columns[i]].mean())
	# In[6]:
	data1.describe()
	# In[7]:
	y=data1['Flood']
	# In[8]:
	for i in range(len(y)):
	    if(y[i] >= 0.1):
	        y[i]=1
	# In[9]:

	y=pd.DataFrame(y)

	data1.drop('Flood',axis=1,inplace=True)


	# In[10]:
	data1.head()
	# In[11]:
	data1.hist(figsize=(6,6));

	#Breaking Date column into timestamp

	d1=pd.DataFrame()
	d1["Day"]=data1['Date']
	d1['Months']=data1['Date']
	d1['Year']=data1['Date']
	data1['Date']=pd.to_datetime(data1['Date'])
	d1["Year"]=data1.Date.dt.year
	d1["Months"]=data1.Date.dt.month
	d1["Day"]=data1.Date.dt.day

	#----------------------Resampling
	#------------not working for piyush
	dx=pd.DataFrame()
	dx['Date']=data1['Date']
	dx['Discharge']=data1['Discharge']
	dx=dx.set_index(['Date'])
	yearly = dx.resample('Y').sum()

	plt.figure(figsize=(9,8))
	plt.xlabel('YEARS')
	plt.ylabel('Level')
	plt.title(filename+" : Year wise Trends")
	plt.plot(yearly,'--')

	#plt.plot(yearly,style=[':', '--', '-'],title='Year wise Trends')
	plt.savefig('static/img/flood.png')
	#--------------------------------


	# In[18]:
	data1.drop('Date',inplace=True,axis=1)
	# In[19]:


	#Scaling the data in range of 0 to 1

	# Scaler=MinMaxScaler(feature_range=(0, 1))
	# Transform=Scaler.fit_transform(data1)
	# # In[20]
	# #Transform
	# # In[21]:
	# Transform=pd.DataFrame(Transform,columns=['Discharge','flood runoff','daily runoff','weekly runoff'])

	# # In[22]:
	# data1=Transform
	# In[23]:
	data1=pd.concat([d1,data1],axis=1)
	data1.head()

	#-----------------------for taking data upto 2015 as training and rest for testing------------------------------------------------
	locate=0;
	for i in range(len(data1["Day"])):
	    if(data1["Day"][i]==31 and data1["Months"][i]==12 and data1["Year"][i]==2015):
	        locate=i;
	        break;
	        
	i=locate+1
	print(i)

	x_train=data1.iloc[0:i,:]
	y_train=y.iloc[0:i]
	x_test=data1.iloc[i:,:]
	y_test=y.iloc[i:]

	x_train.drop(labels=['Day','Months','Year'],inplace=True,axis=1)
	x_test.drop(labels=['Day','Months','Year'],inplace=True,axis=1)


	#-----------------Upsampling the data (as very less entries of flood =1 is present)-----------------
	sm = SMOTE(random_state=2)
	X_train_res, Y_train_res = sm.fit_resample(x_train, y_train)

	x_train, y_train = shuffle( X_train_res, Y_train_res, random_state=0)

	x_train.shape,x_test.shape,y_train.shape,y_test.shape

	# #---------------Logistic Regression--------------------------
	# from sklearn.linear_model import LogisticRegression
	# reg=LogisticRegression()
	# reg.fit(x_train,y_train)
	# y_predict1=reg.predict(x_test)
	# print(set(y_predict1))
	# print(reg.score(x_train,y_train))
	# print(reg.score(x_test,y_test))
	# print(classification_report(y_test, y_predict1))
	# print("mean_absolute_error=",mean_absolute_error(y_test, y_predict1))

   	#------------------------LSTM------------------------------------------------------
	# lstm_size = 27         # 3 times the amount of channels
	# lstm_layers = 2        # Number of layers
	# batch_size = 600       # Batch size
	# seq_len = 1          # Number of steps
	# learning_rate = 0.0001  # Learning rate (default is 0.001)
	# epochs = 1000

	# # Fixed
	# n_classes = 7
	# n_channels = 15
	# graph = tf.Graph()

	# # Construct placeholders
	# with graph.as_default():
    # 	inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    # 	labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    # 	keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    # 	learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')
	# # Convolutional layers
	# with graph.as_default():
    # 	# (batch, 128, 9) --> (batch, 128, 18)
    # 	conv1 = tf.layers.conv1d(inputs=inputs_, filters=30, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu)
    # 	n_ch = n_channels *2
	# with graph.as_default():
    # # Construct the LSTM inputs and LSTM cells
    # lstm_in = tf.transpose(conv1, [1,0,2]) # reshape into (seq_len, batch, channels)
    # lstm_in = tf.reshape(lstm_in, [-1, n_ch]) # Now (seq_len*N, n_channels)
    
    # # To cells
    # lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None) # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?
    
    # # Open up the tensor into a list of seq_len pieces
    # lstm_in = tf.split(lstm_in, seq_len, 0)
    
    # # Add LSTM layers
    # lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_)
    # cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    # initial_state = cell.zero_state(batch_size, tf.float32)
	# with graph.as_default():
    # outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
    #                                                  initial_state = initial_state)
    
    # # We only need the last output tensor to pass into a classifier
    # logits = tf.layers.dense(outputs[-1], n_classes, name='logits')
    
    # # Cost function and optimizer
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    # #optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping
    
    # # Grad clipping
    # train_op = tf.train.AdamOptimizer(learning_rate_)

    # gradients = train_op.compute_gradients(cost)
    # capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    # optimizer = train_op.apply_gradients(capped_gradients)
    
    # # Accuracy
    # correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

	# if (os.path.exists('checkpoints-crnn') == False):
    # !mkdir checkpoints-crnn


	# def get_batches(X, y, batch_size = 100):
	# """ Return a generator for batches """
	# n_batches = len(X) // batch_size
	# X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# # Loop over batches and yield
	# for b in range(0, len(X), batch_size):
	# 	yield X[b:b+batch_size], y[b:b+batch_size]

	#-----------------------LinearDiscriminantAnalysis---------------------------------
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

	clf1=LinearDiscriminantAnalysis()
	clf1.fit(x_train,y_train)

	#-----------------------saving & Loading the model-------------------------------------------

	path='trained/'+filename+'_LDA'
	joblib.dump(clf1, path+'.pkl')
	clf1= joblib.load(path+'.pkl')

	#---------------------------------------------------------------------------------

	y_predict3=clf1.predict(x_test)
	print(set(y_predict3))
	print(clf1.score(x_train,y_train))
	print(clf1.score(x_test,y_test))
	print(classification_report(y_test, y_predict3))
	mae=mean_absolute_error(y_test, y_predict3)
	print("mean_absolute_error=",mae)

	
	#---------------------------KNeighborsClassifier------------------------------------------

	# from sklearn.neighbors import KNeighborsClassifier

	# clf2=KNeighborsClassifier()
	# clf2.fit(x_train,y_train)
	# y_predict4=clf2.predict(x_test)
	# print(set(y_predict4))
	# print(clf2.score(x_train,y_train))
	# print(clf2.score(x_test,y_test))
	# print(classification_report(y_test, y_predict4))
	# print("mean_absolute_error=",mean_absolute_error(y_test, y_predict4))

	# # In[36]:

	#-------------------------------Testing-----------------------------------------------
	# In[38]:
	data1.head()
	# In[39]:
	def predicting(future_data):
		# xx=[13214.0,0.0,0.36,2.08]
		#xx=[4990.0,0.0,1.40,15.38]
		xx=future_data
		xx=np.array(xx)
		xx=xx.reshape((-1, 4))
		xx=clf1.predict(xx)
		# xx=reg.predict(xx)
		return xx
	xx=predicting(fd)
	return xx,mae
#xx=predicted value of flood 0 or 1


