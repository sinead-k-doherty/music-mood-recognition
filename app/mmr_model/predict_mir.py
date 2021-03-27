import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2


def predict_music_mood(song):
    image_size = 128
    num_channels = 3
    songs = []
    classes = ['happy', 'sad', 'angry']

    # Prepare image to input into model
    song = cv2.imread(song)
    song = cv2.resize(song, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    songs.append(song)
    songs = np.array(songs, dtype=np.uint8)
    songs = songs.astype('float32')
    songs = np.multiply(songs, 1.0 / 255.0)

    # Reshape image to the same shape as the layers
    x_batch = songs.reshape(1, image_size, image_size, num_channels)

    # Restore session to recreate TensorFlow graph of MIR model
    sess = tf.Session()
    saver = tf.train.import_meta_graph('app/mmr_model/fir-model4.meta')
    saver.restore(sess, tf.train.latest_checkpoint('app/mmr_model/f'))
    graph = tf.get_default_graph()

    # Retrieve predictions from model
    y_pred = graph.get_tensor_by_name("y_pred:0")

    # Get placeholder variables
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_songs = np.zeros((1, 3))

    # Creating the same dictionary that is used in training the model
    feed_dict_testing = {x: x_batch, y_true: y_test_songs}

    # Get prediction
    prediction = sess.run(y_pred, feed_dict=feed_dict_testing)

    prediction = prediction.tolist()
    print(prediction)
    for index, p in enumerate(prediction[0]):
        prediction[0][index] = float("{0:.4f}".format(p))

    value = max(prediction[0])
    index = prediction[0].index(value)
    emotion = classes[index]
    return str(emotion), str(value)



