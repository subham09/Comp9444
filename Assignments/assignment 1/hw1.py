

import tensorflow as tf

""" PART I """


def add_consts():

    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)
##    sess = tf.Session()
##    print(sess.run(af))
    return af
##add_consts()


def add_consts_with_placeholder():

    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.placeholder(tf.float32)
    a1 = tf.add(c1,c2)
    af = tf.add(a1,c3)
##    sess = tf.Session()
##    print(sess.run(af, feed_dict={c3: 3}))
    return af, c3
#add_consts_with_placeholder()

def my_relu(in_value):

    #sess = tf.Session()
    out_value = tf.maximum(0.0, in_value)
    
    #print(sess.run(out_value, feed_dict={in_value: -3}))
    
    return out_value
#c3 = tf.placeholder(tf.float32)
#x = my_relu(c3)
#sess = tf.Session()
#print(sess.run(x))
def my_perceptron(x):

    #sess = tf.Session()
    pf = tf.placeholder(tf.float32, shape = [x])
    my_int_variable = tf.get_variable("my_int_variable", [x], dtype=tf.float32,initializer=tf.ones_initializer)
    mul = pf*my_int_variable
    #print(sess.run(mul, feed_dict={pf: [1,2,3]}))
    add= tf.reduce_sum(mul)
    out_value = tf.maximum(0.0, add)
    #out_value = my_relu(add)
    
##    sess.run(my_int_variable.initializer)
##    print(sess.run(tf.report_uninitialized_variables()))
    #res_mul = sess.run([mul])
    #print(res_mul)
        
    i =pf
    out = out_value
    return i, out
#my_perceptron(3)

""" PART II """
fc_count = 0  # count of fully connected layers. Do not remove.

def input_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")


def target_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")

def onelayer(X, Y, layersize=10):

    w = tf.Variable(tf.zeros([784, layersize]))
    b = tf.Variable(tf.zeros([layersize]))
    logits1 = tf.matmul(X,w)
    logits = logits1 + b
    preds = tf.nn.softmax(logits)
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = logits)
    batch_loss = tf.reduce_mean(batch_xentropy)

        #out_value = my_relu(add)
    
##    sess.run(my_int_variable.initializer)
##    print(sess.run(tf.report_uninitialized_variables()))
    #res_mul = sess.run([mul])
    #print(res_mul)
    
    return w, b, logits, preds, batch_xentropy, batch_loss


def twolayer(X, Y, hiddensize=30, outputsize=10):

    w1 = tf.Variable(tf.truncated_normal([784, hiddensize]))
    w2 = tf.Variable(tf.zeros([hiddensize, outputsize]))

    b1 = tf.Variable(tf.truncated_normal([hiddensize]))
    b2 = tf.Variable(tf.zeros([outputsize]))

    logitsmul1 = tf.matmul(X, w1)
    logits1 = logitsmul1 + b1
    preds1 = tf.nn.relu(logits1)

    logits12 = tf.matmul(preds1, w2)
    logits = logits12 + b2
    preds = tf.nn.softmax(logits)

    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = logits)
    batch_loss = tf.reduce_mean(batch_xentropy)

        #out_value = my_relu(add)
    
##    sess.run(my_int_variable.initializer)
##    print(sess.run(tf.report_uninitialized_variables()))
    #res_mul = sess.run([mul])
    #print(res_mul)
    
    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss


def convnet(X, Y, convlayer_sizes=[10, 10], \
            filter_shape=[3, 3], outputsize=10, padding="same"):

    my_conv = [filter_shape[0], filter_shape[1], 1, convlayer_sizes[0]]
    weight_conv1 = tf.Variable(tf.truncated_normal(my_conv, stddev=0.1))
    biased_conv1 = tf.Variable(tf.constant(0.1, shape=[convlayer_sizes[0]]))
    conv1 = tf.nn.relu(tf.nn.conv2d(X, weight_conv1, strides=[1, 1, 1, 1], padding='SAME') + biased_conv1)
            
    my_conv2 = [filter_shape[0], filter_shape[1], convlayer_sizes[0], convlayer_sizes[1]]
    weight_conv2 = tf.Variable(tf.truncated_normal(my_conv2, stddev=0.1))
    biased_conv2 = tf.Variable(tf.constant(0.1, shape=[convlayer_sizes[1]]))
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weight_conv2, strides=[1, 1, 1, 1], padding='SAME') + biased_conv2)
            
    X_layer = tf.reshape(conv2, [-1, 28 * 28 * convlayer_sizes[1]])
    w = tf.Variable(tf.zeros([28 * 28 * convlayer_sizes[1], outputsize]))
    b = tf.Variable(tf.zeros([outputsize]))
    logits1 = tf.matmul(X_layer, w)
    logits = logits1 + b
    preds = tf.nn.softmax(logits)
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = logits)
    batch_loss = tf.reduce_mean(batch_xentropy)

        #out_value = my_relu(add)
    
##    sess.run(my_int_variable.initializer)
##    print(sess.run(tf.report_uninitialized_variables()))
    #res_mul = sess.run([mul])
    #print(res_mul)

    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss


def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):

    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary
