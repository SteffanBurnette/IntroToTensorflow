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
#Survived will be our label(output)
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
#Removes the column from dftrain and assigns it to y_train
y_train = dftrain.pop('survived')
#Removes the column from dfeval and assigns it to y_eval
y_eval = dfeval.pop('survived')

#Displays the top 5 values in the dataframe
print(dftrain.head())

#creates a histogram of the 'age' column from the
# dftrain DataFrame, dividing the ages into 20 bins
# to show their distribution. This is commonly used
# in exploratory data analysis to understand the
# spread, central tendency, and shape of the age
# data in the dataset.
dftrain.age.hist(bins=20)
plt.show()

#Counts the number of occurences for each value in the
#specified column and then graphs it
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='barh')
plt.show()

#In summary, this command concatenates the
# dftrain DataFrame and y_train data, groups the
# resulting DataFrame by sex, calculates the mean
# survival rate for each sex, and then plots these
# means as a horizontal bar chart. The x-axis is
# labeled to indicate that it represents the percentage of
# individuals who survived in each sex category. This type
# of analysis and visualization is particularly useful in
# understanding how a categorical variable (like sex) relates
# to a binary outcome (like survival) in a dataset.
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()

#All the columns with categorical data
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
#All the columns with numerical data
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
#Loops through all the columns in the dataframe
for feature_name in CATEGORICAL_COLUMNS:
    #Gets all the unique values from the categorical columns
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
    #.feature_columns is depreciated and will have to use keras for future projects
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

#Breaks the data up into epochs and batchs to feed to the model
#Specifically it will be ten instances of 32 batches feed
#in this case
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
      #passes in a dictonary representation of the dataframe and the lebel
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
#for the evaluation we dont need to shuffle the data
#since were not training it and only need one epoch since were not training it
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# creates an instance of a linear classifier model
# in TensorFlow, specifying which features it should
# use and how to interpret them. This model can then
# be trained and used for classification tasks, part
# icularly where the relationship between features
# and the target variable is expected to be linear.

#Creating the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

#Training the model
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

print(result['accuracy'])  # the result variable is simply a dict of stats about our model