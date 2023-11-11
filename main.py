from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

print(tf.version)

#This is a string tensor variable and how to define one
#Stores one string
#Has a shape of one so it is a scalar
string = tf.Variable("This is a String", tf.string)

#This is a integer tensor variable
integer = tf.Variable(324, tf.int16)

#This is a float tensor variable
flt = tf.Variable(12.34, tf.float64)


# This will be an int32 tensor by default; see "dtypes" below.
#rank_0_tensor = tf.constant(4)
#print(rank_0_tensor)
rank_1_tensor = tf.Variable(["Test"], tf.string)
print(rank_1_tensor)

#Will print out 2 2. The first represents the number of lists and the second represents the amount of elements in the lists
rank2_tensor = tf.Variable([["Test", "okay"], ["Testing", "Not Okay"]], tf.string)
print(rank2_tensor.shape)

#Two lists 3 elements
ones=tf.ones([2,3],dtype=tf.int32)
#two lists, 4 nested lists, 2 elements
#The dimension is 2X4X2 == 16
#To properly reshape we need to make sure that the dimension equals 16 in this instance
twos=tf.ones([2,4,2], dtype=tf.int32)
print("This is the ones")
print(ones)
print("This is the twos")
print(twos)

#Reshaped to 3 list with 2 elements
reshape_ones=tf.reshape(ones, [3,2])
print("This is the reshaped ones")
print(reshape_ones)

#Reshapes to 6 lists of 1 element each
reshape_ones2=tf.reshape(ones, [6, 1])
print("This is the reshaped ones of 6X1")
print(reshape_ones2)

#Reshapes the twos list to a 4X4 == 16
reshape_twos=tf.reshape(twos, [4,4])
print("Reshaped Twos")
print(reshape_twos)


#Creates a zero tensor of 5X5X5X5 ==625
zeros= tf.zeros([5,5,5,5], dtype=tf.int32)
print("This is the zeros tensor of 625")
print(zeros)

###Linear regression
#x is the features and y is the labels (input/output)
x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
#Sets the x and y coordinates
#(0,6)x-axis (0,20)y-axis
plt.axis([0, 6, 0, 20])
plt.show()
#We can see that this data has a linear coorespondence.
# When the x value increases, so does the y.
# Because of this relation we can create a
# line of best fit for this dataset. In this
# example our line will only use one input variable,
# as we are working with two dimensions. In larger
# datasets with more features our line will have more
# features and inputs.

#Titanic data set predicition
# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#Displays the top 5 values in the dataframe
print(dftrain.head())