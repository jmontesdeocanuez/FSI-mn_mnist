# -*- coding: utf-8 -*-

import gzip
import cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


# ---------------- Visualizing some element of the MNIST dataset --------------

# TODO: the neural net!!
train_y = one_hot(train_y, 10)

phImagen = tf.placeholder("float", [None, 784]) #imagen
phLabel = tf.placeholder("float", [None, 10]) #etiqueta

W1 = tf.Variable(np.float32(np.random.rand(784, 15)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(15)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(15, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(phImagen, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(phLabel - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)
error = 99999999
errors = []
epoch = 1
while True:
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={phImagen: batch_xs, phLabel: batch_ys})

    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={phImagen: batch_xs, phLabel: batch_ys})
    result = sess.run(y, feed_dict={phImagen: batch_xs})
    errorPrev = error
    error = sess.run(loss, feed_dict={phImagen: valid_x, phLabel: valid_y})
    errors.append(error)
    print "VALIDACION -> Epoch #:", epoch, "Error: ", error

    if(abs(error-errorPrev)/errorPrev) < 0.001 or error > errorPrev:
        break
    epoch = epoch + 1

test_result = sess.run(y, feed_dict={phImagen: test_x})
aciertos = 0.
for calculado, real in zip(test_result, test_y):
    if (np.argmax(calculado) == np.argmax(real)):
        aciertos = aciertos+1

tasaAciertos = (aciertos/len(test_result))*100

print "Tasa de aciertos al probar con el conjunto de test:", tasaAciertos,"%"

plt.plot(errors)
plt.xlabel("Numero de epocas")
plt.ylabel("Error de validacion")
plt.title("Evolucion del error durante el entrenamiento")
plt.show()