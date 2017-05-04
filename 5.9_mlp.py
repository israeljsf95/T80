# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 23:58:22 2016

@author: Israel
Modulo de Exercicio pratico 5.9 da Livro de
Redes Neurais do Ivan Nunes

"""

# Third-party libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def carrega_data():
  file = open('../dados/Tabela5.9.txt',)
  x = np.loadtxt(file, skiprows = 1, usecols = (0, 1, 2, 3,))
  x = np.array(x)
  file.seek(0,0)
  y = np.loadtxt(file, skiprows = 1, usecols=(4, 5, 6,))
  y = np.array(y)
  file = open('../dados/Tabela5.9t.txt',)
  tx = np.loadtxt(file, skiprows = 1, usecols = (0,1,2,3,))
  tx = np.array(tx)
  file.seek(0,0)
  ty = np.loadtxt(file, skiprows = 1, usecols=(4, 5, 6,))
  ty = np.array(ty)
  return (x, y, tx, ty)


x = tf.placeholder(tf.float32, shape = [None, 4], name = "x")
y = tf.placeholder(tf.float32, shape = [None, 3], name = "y")
teta1 = tf.Variable(tf.random_uniform([4,15], -1, 1), name = 'teta1')
teta2 = tf.Variable(tf.random_uniform([15,3], -1, 1), name = 'teta2')
bias1 = tf.Variable(tf.ones([15]), name = 'bias1')
bias2 = tf.Variable(tf.ones([3]), name = 'bias2')

#Primeiro Produto 
with tf.name_scope("layer2") as scope:
  y1 = tf.sigmoid(tf.matmul(x, teta1) + bias1)

#Segundo Produto
with tf.name_scope("layer3") as scope:
  y2 = tf.sigmoid(tf.matmul(y1, teta2) + bias2)

with tf.name_scope("cost") as scope:
  eta = 0.1
  cost = tf.reduce_mean(tf.square(y2 - y))
                                           
with tf.name_scope("train") as scope:
  train_step = tf.train.MomentumOptimizer(0.1, 0.9).minimize(cost)

#data set
x_, y_, xt_, yt_= carrega_data()
cost_ = []
acc_ = []
#inicializando o processo 
model = tf.global_variables_initializer()
sess = tf.Session()
sess.run(model)
for i in range(6000):
  cost_.append(sess.run(cost, feed_dict = {x: x_, y: y_}))
  sess.run(train_step, feed_dict = {x: x_, y: y_}) 
  if i%100 == 0:
    print('Epoch: ', i)
    print('Y2: \n', sess.run(y2, feed_dict = {x: x_, y: y_}))
    print('Teta1: ', sess.run(teta1))
    print('Bias1: ', sess.run(bias1))
    print('Teta2: ', sess.run(teta2))
    print('Bias2: ', sess.run(bias2))
    print('Custo: ', sess.run(cost, feed_dict = {x: x_, y: y_}))

print("Y2\n", sess.run(y2, feed_dict = {x: xt_, y: yt_}))
print("Y2\n", np.around(sess.run(y2, feed_dict = {x: xt_, y: yt_})))
plt.xlabel('Epoca')
plt.ylabel('Custo')
plt.plot(cost_)
plt.show()

