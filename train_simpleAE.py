from __future__ import division, print_function, absolute_import

import tensorflow as tf
from model_simpleAE import AutoEncoder
from image_utilities import ImageUtils


process_and_save_images = True
flatten_before_encode = True
img_shape = (50,50)
seed = 100
num_steps = 20000
batch_size = 10
img_train_dir_raw = "train_raw"
img_train_dir_proc = "train_proc"
img_test_dir_raw = "test_raw"
img_test_dir_proc = "test_proc"


IU = ImageUtils()

# Process and save
if process_and_save_images:

    # Training images
    IU.raw2resized_load_save(raw_dir=img_train_dir_raw,
                             processed_dir=img_train_dir_proc,
                             img_shape=img_shape)


AE = AutoEncoder()
with tf.Session() as sess:

    # Run the initializer
    sess.run(AE.init)
    saver = tf.train.Saver()
    x_data_train, all_train_filenames = IU.raw2resizednorm_load(img_dir=img_train_dir_proc, img_shape=img_shape)
    print("x_data_train.shape = {0}".format(x_data_train.shape))
    # Flatten data if necessary
    if flatten_before_encode:
        x_data_train = IU.flatten_img_data(x_data_train)
    print("x_data_train.shape = {0}".format(x_data_train.shape))
    # Training
    for i in range(1, num_steps + 1):
        training_steps = int((len(x_data_train) / batch_size)) + 1
        for step in range(training_steps):
            X_batch = x_data_train[(step * batch_size):((step + 1) * batch_size)]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([AE.optimizer, AE.loss], feed_dict={AE.X: X_batch})
        # Display loss per step
        print('Step %i: Minibatch Loss: %f' % (i, l))
    save_path = saver.save(sess, "model/model_simpleAE.ckpt")

