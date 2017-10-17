import tensorflow as tf

#the regression function used to refine the neural net
def optimize(graph, categories):
    y_ = tf.placeholder(tf.float32, [None, categories])
    return y_, tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=graph))

#train the neural net that is passed in
def train(loss_fn):
    return tf.train.AdamOptimizer(1e-4).minimize(loss_fn)

#used to determine the prediction of the neural net, and then validate that
#prediction
def correct_prediction(y, y_):
    return tf.nn.in_top_k(y, y_, 1)
