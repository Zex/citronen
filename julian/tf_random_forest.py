#
import os, sys, glob
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.contrib import learn, layers, framework
from tensorflow.contrib.tensor_forest.python import tensor_forest
from with_tf import Julian, start


class Forest(Julian):

    def __init__(self, args):
        super(Forest, self).__init__(self)
        self.total_class = 285513 

    def _build_model(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.seqlen], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")

        params = tensor_forest.ForestHParams(
            num_classes=self.total_class,
            num_trees=100,
            max_nodes=10000000,
            num_features=10000,
            ).fill()
        graph = tensor_forest.RandomForestGraphs(params)
        self.train_op = graph.training_graph(self.input_x, self.input_y)
        self.loss = graph.loss_graph(self.input_x, self.input_y)
        self.pred = graph.inference_graph(self.input_x)
        self.acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.pred, 1),
                tf.cast(self.input_y, tf.int64)), tf.float32))

        summary.append(tf.summary.scalar("loss", self.loss))
        summary.append(tf.summary.scalar("acc", self.acc))

        self.summary = tf.summary.merge(summary, name="merge_summary")

        
if __name__ == '__main__':
    start()
