import tensorflow as tf
import tensorlayer as tl
import numpy as np
import cv2
# from vggnet.imagenet_classes import *
from tfmyrcnn.imagenet_classes import *
import os
import tflearn.datasets.oxflower17 as oxflower17
Xin, Y = oxflower17.load_data(one_hot=True, resize_pics=(224, 224))
keep_prob=tf.placeholder(tf.float32)
def conv_layer(net_in):
    with tf.name_scope('preprocess') as scope:
        """
        Notice that we include a preprocessing layer that takes the RGB image
        with pixels values in the range of 0-255 and subtracts the mean image
        values (calculated over the entire ImageNet training set).
        """
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean
    """ conv1 """

    network=tl.layers.Conv2dLayer(
        net_in,act=tf.nn.relu,
        shape=[3,3,3,64],
        strides=[1,1,1,1],
        padding='SAME',
        name="conv1_1",
        W_init_args={'trainable':False},
        b_init_args={'trainable':False}
    )
    network=tl.layers.Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3,3,64,64],
        strides=[1,1,1,1],
        padding='SAME',
        name='conv1_2',
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    """pool1"""
    network=tl.layers.PoolLayer(
        network,
        ksize=[1,2,2,1],
        strides=[1,2,2,1],
        padding='SAME',
        pool=tf.nn.max_pool,
        name='pool1'
    )

    """conv2"""
    network = tl.layers.Conv2dLayer(
        network, act=tf.nn.relu,
        shape=[3, 3, 64, 128],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name="conv2_1",
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    network = tl.layers.Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 128, 128],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv2_2',
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    """pool2"""
    network = tl.layers.PoolLayer(
        network,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        pool=tf.nn.max_pool,
        name='pool2'
    )

    """conv3"""
    network = tl.layers.Conv2dLayer(
        network, act=tf.nn.relu,
        shape=[3, 3, 128, 256],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name="conv3_1",
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    network = tl.layers.Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 256, 256],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv3_2',
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    network = tl.layers.Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 256, 256],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv3_3',
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    """pool3"""
    network = tl.layers.PoolLayer(
        network,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        pool=tf.nn.max_pool,
        name='pool3'
    )

    """conv4"""
    network = tl.layers.Conv2dLayer(
        network, act=tf.nn.relu,
        shape=[3, 3, 256, 512],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name="conv4_1",
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    network = tl.layers.Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 512],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv4_2',
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    network = tl.layers.Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 512],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv4_3',
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    """pool4"""
    network = tl.layers.PoolLayer(
        network,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        pool=tf.nn.max_pool,
        name='pool4'
    )
    """conv5"""
    network = tl.layers.Conv2dLayer(
        network, act=tf.nn.relu,
        shape=[3, 3, 512, 512],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name="conv5_1",
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    network = tl.layers.Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 512],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_2',
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    network = tl.layers.Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 512],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_3',
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    """pool5"""
    network = tl.layers.PoolLayer(
        network,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        pool=tf.nn.max_pool,
        name='pool5',

    )
    return network

def fc_layer(network):
    network=tl.layers.FlattenLayer(network,name="flatten")

    network=tl.layers.DenseLayer(
        network,
        n_units=4096,
        act=tf.nn.relu,
        name="fc1_relu",
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    network=tl.layers.DropoutLayer(network)
    network = tl.layers.DenseLayer(
        network,
        n_units=4096,
        act=tf.nn.relu,
        name="fc2_relu",
        W_init_args={'trainable': False},
        b_init_args={'trainable': False}
    )
    network = tl.layers.DropoutLayer(network,name="drp2")

    network = tl.layers.DenseLayer(
        network,
        n_units=1000,
        act=tf.identity,
        name="fc3"
    )
    # network=tl.layers.DropoutLayer(network,name="dropout3")
    return network

if __name__ == "__main__":

    sess = tf.InteractiveSession()

    # input_data op
    X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    y_ = tf.placeholder(dtype=tf.int32, shape=[None, 17], name="y_")
    net_in = tl.layers.InputLayer(X, name="input_layer")

    # create network op
    net_cnn = conv_layer(net_in)
    network = fc_layer(net_cnn)
    y = network.outputs
    prob = tf.nn.softmax(y)
    y_op = tf.argmax(prob, 1)
    print(y.get_shape())
    y_truth = tf.argmax(y_, 1)


    # cost op
    cost = tl.cost.cross_entropy(y, y_truth)

    # predict and accuracy op
    correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_truth, tf.float32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train op
    # train=tf.train.AdamOptimizer(0.001).minimize(cost)
    # train op momentu
    train = tf.train.MomentumOptimizer(0.001,0.9).minimize(cost)

    # initilize op
    tl.layers.initialize_global_variables(sess)

    # print op
    network.print_layers()
    network.print_params()

    ### saver
    # saver = tf.train.Saver()
    # saver.restore(sess, "./vggnet.model")
    # print("load done")
    ##### end

    ##
    ## using npz file to load
    npz = np.load('/home/cheku/weights/vgg16_weights.npz')

    params = []
    for val in sorted(npz.items()):

        if val[0]=="fc8_W" or val[0]=="fc8_b":
            continue

        print( "loadding %s"%(val[0]))
        print("  Loading %s" % str(val[1].shape))
        params.append(val[1])

    tl.files.assign_params(sess, params, network)

    Xin*=255.0
    Xin-=[123.68, 116.779, 103.939]
    # simple split the data op
    rate=0.1
    length=np.shape(Xin)[0]
    index=list(range(length))
    np.random.shuffle(index)
    d=int(np.floor(rate*length))
    train_data_X=Xin[index[d:],]
    train_data_Y=Y[index[d:],]
    test_data_X=Xin[index[0:d],]
    test_data_Y=Y[index[0:d],]
    #
    # #

    # define some params op
    BATCH_SIZE=64
    n_epoch=50
    n_iterators=int(np.floor(len(train_data_X)/BATCH_SIZE))

    for i in range(n_epoch):
        for iter_num in range(n_iterators):

            data_X = train_data_X[(iter_num * BATCH_SIZE):((iter_num + 1) * BATCH_SIZE), ]
            data_Y = train_data_Y[(iter_num * BATCH_SIZE):((iter_num + 1) * BATCH_SIZE), ]
            feed_dict = {X: data_X, y_: data_Y}
            feed_dict.update(network.all_drop)
            sess.run(train, feed_dict=feed_dict)

            dp_dict = tl.utils.dict_to_one(network.all_drop)
            feed_dict = {X: data_X, y_: data_Y}
            feed_dict.update(dp_dict)
            cost_value = sess.run(cost, feed_dict=feed_dict)
            accuracy = sess.run(acc, feed_dict=feed_dict)
            # print(sess.run(network.all_params[28][1,1:5]))
            print("the epoch is %d ,the iter is %d and the cost is %lf and the accuaracy is %lf "%
                  (i, iter_num, cost_value,accuracy ))

        data_X = train_data_X[(iter_num * BATCH_SIZE):, ]
        data_Y = train_data_Y[(iter_num * BATCH_SIZE):, ]
        feed_dict = {X: data_X, y_: data_Y}
        feed_dict.update(network.all_drop)
        sess.run(train, feed_dict=feed_dict)

        dp_dict = tl.utils.dict_to_one(network.all_drop)
        feed_dict = {X: test_data_X, y_: test_data_Y}
        feed_dict.update(dp_dict)
        cost_value = sess.run(cost, feed_dict=feed_dict)
        accuracy=sess.run(acc,feed_dict=feed_dict)
        print("epecho done ---------------------------------------")
        print("the cost is %lf and the accuracy is %lf "%(cost_value,accuracy))
        print("next one -------------------------------")




