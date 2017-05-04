# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 23:58:22 2016

@author: Israel
Modulo de Exercicio pratico 5.8 da Livro de
Redes Neurais do Ivan Nunes

"""

# Third-party libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def carrega_data():
    file = open('../dados/Tabela5.txt', )
    x = np.loadtxt(file, skiprows = 1, usecols = (0,1,2,))
    x = np.array(x)
    file.seek(0,0)
    y = np.loadtxt(file, skiprows = 1, usecols=(3,))
    y = np.array(y)
    y.shape = (200,1)
    file.close()
    file = open('../dados/testd5.txt', )
    tx = np.loadtxt(file, usecols = (0,1,2))
    tx = np.array(tx)
    file.seek(0,0)
    ty = np.loadtxt(file, usecols=(3,))
    ty = np.array(ty)
    ty.shape = (20,1)
    return (x, y, tx, ty)

x = tf.placeholder(tf.float32, shape = [None, 3], name = "x")
y = tf.placeholder(tf.float32, shape = [None, 1], name = "y")
teta1 = tf.Variable(tf.random_uniform([3,10], -1, 1), name = 'teta1')
teta2 = tf.Variable(tf.random_uniform([10,1], -1, 1), name = 'teta2')
bias1 = tf.Variable(tf.zeros([10]), name = 'bias1')
bias2 = tf.Variable(tf.zeros([1]), name = 'bias2')

#Primeiro Produto 
with tf.name_scope("layer2") as scope:
  y1 = tf.sigmoid(tf.matmul(x, teta1) + bias1)

#Segundo Produto
with tf.name_scope("layer3") as scope:
  y2 = tf.sigmoid(tf.matmul(y1, teta2) + bias2)

with tf.name_scope("cost") as scope:
  eta = 0.02
  cost = tf.reduce_mean(tf.square(y2 - y) + eta*tf.nn.l2_loss(teta1)
                                          + eta*tf.nn.l2_loss(bias1)
                                          + eta*tf.nn.l2_loss(teta2)
                                          + eta*tf.nn.l2_loss(bias2)) 
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


#data set
x_, y_, xt_, yt_ = carrega_data()
cost_ = []
#inicializando o processo 
model = tf.global_variables_initializer()
sess = tf.Session()
sess.run(model)
for i in range(5000):
  cost_.append(sess.run(cost, feed_dict = {x: x_, y: y_}))
  sess.run(train_step, feed_dict = {x: x_, y: y_}) 
  if i%100 == 0:
    print('Epoch: ', i)
    print('Y2: ', sess.run(y2, feed_dict = {x: x_, y: y_}))
    print('Teta1: ', sess.run(teta1))
    print('Bias1: ', sess.run(bias1))
    print('Teta2: ', sess.run(teta2))
    print('Bias2: ', sess.run(bias2))
    print('Custo: ', sess.run(cost, feed_dict = {x: x_, y: y_}))

#Validacao da Entrada
print('Y2: ', sess.run(y2, feed_dict = {x: xt_, y: yt_}))
plt.xlabel('Epoca')
plt.ylabel('Custo')
plt.plot(cost_)
plt.show()