#!/usr/bin/env python2.7

import numpy as np
import sklearn.datasets as datasets
import tensorflow as tf
import argparse

def _float64_feature(feature):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[feature]))

def _feature_dict_from_row(row):
    """Take row of length n+1 from 2-D ndarray and convert it to a dictionary:

      {
        'f0': row[0],
        'f1': row[1],
         ...
        'fn': row[n]
      }
    """

    features = {}
    # TODO(davidr): switch to a dict comprehension when you're sure this does what
    #               you think it does
    for i in range(len(row)):
        features["f{:d}".format(i)] = _float64_feature(row[i])
    return features

def gen_regression_dataset(n_samples, n_features):
    # We'll say 50% of the features are informative
    n_informative = int(n_features * .5)

    X, y, intercept = datasets.make_regression(n_samples=n_samples,
                                               n_features=n_features,
                                               n_informative=n_informative,
                                               n_targets=1,
                                               noise=0.1,
                                               bias=np.pi,
                                               coef=True,
                                               random_state=31337 # For teh reproducibilities
                                                                  # (and lulz)
                                               )
    # Don't need the intercept yet
    return X, y

def write_regression_data_to_tfrecord(X, y, filename):
    with tf.python_io.TFRecordWriter('{:s}.tfrecords'.format(filename)) as tfwriter:
        for row_index in range(X.shape[0]):

            # For 5000x100, this takes about 180s. That seems long, doesn't it?
            # TODO(davidr): I think the delay is in _feature_dict_from_row()
            # write_data_to_tfrecord(X, y)
            features = _feature_dict_from_row(X[row_index])

            # We only generated the feature_dict from the feature matrix. Tack on the label
            features['label'] = _float64_feature(y[row_index])

            example = tf.train.Example(features=tf.train.Features(feature=features))
            tfwriter.write(example.SerializeToString())

def tf_recordtest(args):
    X, y = gen_regression_dataset(n_samples=args.n_samples, n_features=args.n_features)
    write_regression_data_to_tfrecord(X, y, args.filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="write some random tfrecords")
    parser.add_argument('-s', '--samples', metavar='N', type=int, default=500, dest='n_samples',
                        help='Number of samples/observations')
    parser.add_argument('-f', '--features', metavar='M', type=int, default=10, dest='n_features',
                        help='Number of features/targets')
    parser.add_argument('-F', '--filename', metavar="FILE", dest='filename', required=True,
                        help='Filename for tfrecord output (will append .tfrecords)')
    args = parser.parse_args()

    tf_recordtest(args)
