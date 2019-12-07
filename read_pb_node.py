import tensorflow as tf

with tf.Session() as sess:
    with tf.gfile.FastGFile('./model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        print(graph_def.node)
