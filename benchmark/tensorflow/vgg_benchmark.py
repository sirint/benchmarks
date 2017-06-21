import tensorflow as tf

import vgg19_trainable as vgg19
import numpy as np
import time

img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger

batch = np.random.rand(1, 224, 224, 3).astype(np.float32)

with tf.device('/gpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 1000])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19()
    vgg.build(images, train_mode)

    sess.run(tf.global_variables_initializer())

    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    print 'Start'
    iters = 200
    start = time.time()
    for i in range(iters):
        st = time.time()
        sess.run(train, feed_dict={images: batch, true_out: [img1_true_result], train_mode: False})
        e = time.time()
        print i + 1,': iteration time {}'.format(e - st)

    end = time.time()
    print('Time : {}'.format(end - start))
