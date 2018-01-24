import image_utilities
from model_simpleAE import AutoEncoder
import tensorflow as tf
from math import *
import operator
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

img_train_dir_proc = "train_proc"


#def square_rooted(x):
#    return round(sqrt(sum([a * a for a in x])), 3)


#def cosine_similarity(x, y):
#    numerator = sum(a * b for a, b in zip(x, y))
#    denominator = square_rooted(x) * square_rooted(y)
#    return round(numerator / float(denominator), 3)


def model_similarity(model, image, shape):
    IU =  image_utilities.ImageUtils()
    x_data_train, all_train_filenames = IU.raw2resizednorm_load(img_dir= img_train_dir_proc, img_shape=shape)
    x_data_test = IU.img_resizednorm_load(img_path= image, img_shape=shape)
    x_data_test = IU.flatten_img_data(x_data_test)
    x_data_train = IU.flatten_img_data(x_data_train)
    AE = AutoEncoder()
    with tf.Session() as sess:
        sess.run(AE.init)
        saver = tf.train.import_meta_graph(model+".meta")
        saver.restore(sess, model)

        embedding_dict = {}

        for x, y in zip(x_data_train, all_train_filenames):
            g_train = sess.run(AE.encoder_op, feed_dict={AE.X: [x]})
            embedding_dict.update({y: g_train})

        g_test = sess.run(AE.encoder_op, feed_dict={AE.X: x_data_test})

        similarity = {}
        for y_key, y_value in embedding_dict.iteritems():
            similarity.update({y_key:cosine_similarity([g_test[0]], [y_value[0]])})
            #similarity.update({y_key: cosine_similarity(g_test[0], y_value[0])})
        print(sorted(similarity.items(), key=operator.itemgetter(1))[0:5])


model_similarity("model/model_simpleAE.ckpt","test_raw/fries2_resized.jpeg", (50,50))

