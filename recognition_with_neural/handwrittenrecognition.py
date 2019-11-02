#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy
import scipy.special
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# class neural network dengan spesifikasi 3 buah layer.
class neuralNetwork:
    
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # Bobot diantara dua buah layer yaitu layer1 dan layer2 lalu layer2 dan layer3
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        # learning rate
        self.lr = learningrate
        # fungsi aktivasi menggunakan fungsi sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    
    # train neural network
    def train(self, inputs_list, targets_list):
        # mengubah input dari array sequens menjadi matriks
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # Menghitung nilai nodes dari nilai input dan bobotnya
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Menggunakan fungsi aktivasi pada nilai nodes
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Menghitung nilai nodes dari nodes hidden ke nodes output
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Menggunakan fungsi aktivasi pada nodes output
        final_outputs = self.activation_function(final_inputs)
        
        # Menghitung nilai eror pada nilai nodes output dari hasil forwardpass neural
        output_errors = targets - final_outputs
        # Menghitung nilai eror pada nilai nodes hidden dari hasil forwardpass neural
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # Update bobot antara layer hidden dan layer output
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # Update bobot antara layer input dan layer hidden
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # Query prediksi
    def query(self, inputs_list):
        # mengubah input dari array squens menjadi matriks
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # Menghitung nilai antara layer input dan hidden
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Menghitung nilai antara layer hidden dan output
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# In[3]:


# 784 merupakan nilai piksel pada gambar
input_nodes = 784
# jumlah hidden nodes merupakan observasi
hidden_nodes = 200
# jumlah output nodes merupakan nilai label prediksi
output_nodes = 10

# learning rate diset 0.1
learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# In[4]:


# Data tulisan tangan 0-1 dari mnist
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# In[5]:


# epoch bergantung dari tingkat model untuk mencapai konvergen, semakin lama maka dibutuhkan epoch yang lebih bayak lagi
epochs = 5

for e in range(epochs):
    for record in training_data_list:
        # delimitersplit nilai pada csv adalah tanda ,
        all_values = record.split(',')
        # preprocessing data
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # membuat matriks label
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        
        n.train(inputs, targets)
        pass
    pass


# In[6]:


# Data test mnist
test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# In[7]:


# Testing data
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    
    pass


# In[8]:


# Hasil performansi
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

