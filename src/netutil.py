import tensorflow as tf

def optimize(graph, categories):
    y_ = tf.placeholder(tf.float32, [None, categories])
    return y_, tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=graph)

def correct_prediction(y, y_):
    return tf.nn.in_top_k(y, y_, 1)
