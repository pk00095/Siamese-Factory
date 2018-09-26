import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2 as gpb
from google.protobuf import text_format

with tf.Session() as sess:
    model_filename ='siam.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())   # to read from .pb file
    #graph_def = gpb.GraphDef()
    #with open(model_filename,'r') as f:
     #      data=f.read()
    #text_format.Parse(data, graph_def) # to read from .pbtxt file
        g_in = tf.import_graph_def(graph_def)
LOGDIR='./logs/'
if not tf.gfile.exists(LOGDIR):
    tf.gfile.MkDir(LOGDIR)
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()

