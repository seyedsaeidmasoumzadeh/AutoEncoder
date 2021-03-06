import image_utilities
from model_simpleAE import AutoEncoder
import tensorflow as tf
import operator
from sklearn.metrics.pairwise import cosine_similarity

img_train_dir_proc = "train_proc"


def model_similarity(model, image, shape):
    IU =  image_utilities.ImageUtils()
    x_data_train, all_train_filenames = IU.load_raw2resizednorm(img_dir= img_train_dir_proc, img_shape=shape)
    x_data_test = IU.load_img_resizednorm(img_path= image, img_shape=shape)
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
        for x in sorted(similarity.items(), key=operator.itemgetter(1), reverse=True)[:5]:
            print(x)

# Find 5 similar images to a given image(please select your desire model,  SimpleAE or ConvAE)
model_similarity("model/model_convAE.ckpt","test_raw/salad1_resized.jpeg", (50,50))

