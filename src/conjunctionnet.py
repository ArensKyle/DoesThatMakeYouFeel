import tensorflow as tf
import conjunction

def create_graph(categories):
    cL = len(conjunction.CONJUNCTIONS)

    L = tf.placeholder(tf.float32, [None, categories])
    R = tf.placeholder(tf.float32, [None, categories])
    c = tf.placeholder(tf.float32, [None, cL])
    
    LW = tf.Variable(tf.zeros([cL, categories]))
    RW = tf.Variable(tf.zeros([cL, categories]))
    b = tf.Variable(tf.zeros([catagories]))

    LWFilt = tf.matmul(c, LW)
    RWFilt = tf.matmul(c, LW)

    Sum = tf.multiply(L, LWFilt) + tf.multiply(R, RWFilt)

    return tf.nn.softmax(tf.nn.bias_add(Sum, b))
