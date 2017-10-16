import tensorflow as tf
import conjunction

def create_graph(categories):
    c = len(conjunction.CONJUNCTIONS)

    #left and right side category values (e.g. positive, negative)
    left_side = tf.placeholder(tf.float32, [None, categories])
    right_side = tf.placeholder(tf.float32, [None, categories])
    #selected conjunction
    conjuction = tf.placeholder(tf.float32, [None, c])

    #left and right side weights for each conjunction
    left_weights = tf.Variable(tf.zeros([c, categories]))
    right_weights = tf.Variable(tf.zeros([c, categories]))
    bias = tf.Variable(tf.zeros([categories]))

    #scalars for left and right side wieghts based on the selected conjunction
    left_scalar = tf.matmul(c, left_weights)
    right_scalar = tf.matmul(c, right_weights)

    #combine the left and right sides
    result = tf.multiply(left_side, left_scalar) + tf.multiply(right_side, right_scalar)

    #apply bias when returning
    return tf.nn.softmax(tf.nn.bias_add(result, bias))
