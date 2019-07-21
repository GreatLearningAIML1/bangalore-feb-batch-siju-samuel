#!/usr/bin/env python
# coding: utf-8

# ### Load tensorFlow

# In[1]:


#!pip install tensorflow


# In[4]:


import tensorflow as tf


# In[5]:


#Reset Default graph - Needed only for Jupyter notebook
tf.reset_default_graph()


# In[4]:


#from google.colab import drive
#drive.mount('/content/drive')
#/content/drive/My Drive/ANN Mahesh Anand/


# ### Collect Data

# In[1]:


from tensorflow.python.keras.datasets import boston_housing

#Load data
(features, actual_prices),_ = boston_housing.load_data(test_split=0)


# In[2]:


print('Number of examples: ', features.shape[0])
print('Number of features for each example: ', features.shape[1])
print('Shape of actual prices data: ', actual_prices.shape)


# # Building the graph

# Define input data placeholders

# In[6]:


#Input features
x = tf.placeholder(shape=[None,13],dtype=tf.float32, name='x-input')

#Normalize the data
x_n = tf.nn.l2_normalize(x,1)

#Actual Prices
y_ = tf.placeholder(shape=[None],dtype=tf.float32, name='y-input')


# Define Weights and Bias

# In[7]:


W = tf.Variable(tf.zeros(shape=[13,1]), name="Weights")
b = tf.Variable(tf.zeros(shape=[1]),name="Bias")


# Prediction

# In[8]:


#We will use normalized data
#y = tf.add(tf.matmul(x,W),b,name='output')
y = tf.add(tf.matmul(x_n,W),b,name='output')


# Loss (Cost) Function

# In[9]:


loss = tf.reduce_mean(tf.square(y-y_),name='Loss')


# GradientDescent Optimizer to minimize Loss

# In[10]:


train_op = tf.train.GradientDescentOptimizer(0.03).minimize(loss)


# # Executing the Graph

# In[1]:


#Lets start graph Execution
sess = tf.Session()

# variables need to be initialized before we can use them
sess.run(tf.global_variables_initializer())

#how many times data need to be shown to model
training_epochs = 100


# In[ ]:


for epoch in range(training_epochs):
            
    #Calculate train_op and loss
    _, train_loss = sess.run([train_op,loss],feed_dict={x:features, y_:actual_prices})
    
    if epoch % 10 == 0:
        print ('Training loss at step: ', epoch, ' is ', train_loss)


# In[ ]:


sess.close()

