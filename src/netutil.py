import tensorflow as tf

def optimize(graph, categories):
    y_ = tf.placeholder(tf.float32, [None, categories])
    return y_, tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=graph))

def train(loss_fn):
    return tf.train.AdamOptimizer(1e-4).minimize(loss_fn)

def correct_prediction(y, y_):
    return tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
