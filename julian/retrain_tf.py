import tensorflow as tf
import os
import glob

graph_path = sorted(glob.glob("../models/springer/*meta"), key=os.path.getmtime)[-1] 
model_dir = "../models/springer"

class hello(object):

    pass

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(graph_path)
    saver.restore(sess,tf.train.latest_checkpoint(model_dir))

    graph = tf.get_default_graph()
    print(dir(graph))

    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    print(global_vars)
    global_vars = sess.run(tf.global_variables())
    
    x = graph.get_tensor_by_name("input_x:0")
    print(x)
    print(graph.get_operations())

#w1 = graph.get_tensor_by_name("w1:0")
#w2 = graph.get_tensor_by_name("w2:0")
#feed_dict ={w1:13.0,w2:17.0}

#op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
#print sess.run(op_to_restore,feed_dict)
    h = hello()
    names = []
    for v in tf.global_variables():
        if v in tf.trainable_variables():
            vname = variable_averages.average_name(v)
            print("train", vname)
        else:
            vname = v.op.name
            print("non-train", vname)
        setattr(h, vname, v)
        names.append(vname)

    for n in names:
        print(getattr(h, n))

    summary = tf.get_collection(tf.GraphKeys.SUMMARIES)
    for x in summary:
        print(x)


    print(sess.run(graph.get_tensor_by_name("global_step:0")))
    print(graph.get_tensor_by_name("train_op:0").op)
