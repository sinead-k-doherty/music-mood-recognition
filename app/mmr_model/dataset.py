import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


class Dataset(object):

    def __init__(self, data, labels, data_names, classes):
        self._num_examples = data.shape[0]
        self._data = data
        self._labels = labels
        self._data_names = data_names
        self._classes = classes
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def data_names(self):
        return self._data_names

    @property
    def classes(self):
        return self._classes

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._data[start:end], self._labels[start:end], self._data_names[start:end], self._classes[start:end]


def load_train_music(fields):
    songs = []
    labels = []
    song_names = []
    classes = []

    for field in fields:
        index = fields.index(field)
        song_files = glob.glob('/mmr_model/music/spectrograms/' + field + '/*')

        for file in song_files:
            song = cv2.imread(file)
            song = cv2.resize(song, (128, 128), 0, 0, cv2.INTER_LINEAR)
            song = song.astype(np.float32)
            song = np.multiply(song, 1.0 / 255.0)
            songs.append(song)
            label = np.zeros(len(fields))
            label[index] = 1.0
            labels.append(label)
            song_name = os.path.basename(file)
            song_names.append(song_name)
            classes.append(field)

    songs = np.array(songs)
    labels = np.array(labels)
    song_names = np.array(song_names)
    classes = np.array(classes)
    return songs, labels, song_names, classes


def read_train_sets(data_size, classes, data_type):
    class Datasets(object):
        pass
    datasets = Datasets()
    data, labels, data_names, cls = load_train_music(classes)
    data, labels, data_names, cls = shuffle(data, labels, data_names, cls)

    validation_size = int(0.2 * data.shape[0])
    validation_data = data[:validation_size]
    validation_labels = labels[:validation_size]
    validation_data_names = data_names[:validation_size]
    validation_classes = cls[:validation_size]

    training_data = data[validation_size:]
    training_labels = labels[validation_size:]
    training_data_names = data_names[validation_size:]
    training_classes = cls[validation_size:]

    datasets.train = Dataset(training_data, training_labels, training_data_names, training_classes)
    datasets.valid = Dataset(validation_data, validation_labels, validation_data_names, validation_classes)
    return datasets


