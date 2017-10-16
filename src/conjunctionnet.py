import tensorflow as tf
import conjunction

def create_graph(categories):
    cL = len(conjunction.CONJUNCTIONS)

    L = tf.placeholder(tf.float32, [None, categories])
    R = tf.placeholder(tf.float32, [None, categories])
    c = tf.placeholder(tf.float32, [None, cL])
    
    LW = tf.Variable(tf.zeros([catagories, cL]))
    RW = tf.Variable(tf.zeros([catagories, cL]))
    b = tf.Variable(tf.zeros([catagories]))

    LWFilt = tf.matmul(LW, c, transpose_b=True)
    RWFilt = tf.matmul(RW, c, transpose_b=True)

    Sum = tf.matmul(L, LWFilt) + tf.matmul(R,RWFilt) + b

    return tf.nn.softmax(Sum)
