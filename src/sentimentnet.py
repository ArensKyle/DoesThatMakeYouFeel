import tensorflow as tf

TWEET_WL_MAX = 70
EMBEDDING_SIZE = 64
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 10

def create_graph(categories, vocab_size, feature_size):
    tweet_w_input = tf.placeholder(tf.int32, [None, TWEET_WL_MAX])
    tweet_f_input = tf.placeholder(tf.float32, [None, TWEET_WL_MAX, feature_size])
    # add dropout?

    # EMBEDDING
    W = tf.Variable(
            tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0))
    embedded_chars = tf.nn.embedding_lookup(W, tweet_w_input)
    embedded_expanded = tf.expand_dims(embedded_chars, -1)

    # Feature adjustment
    W_feat = tf.Variable(tf.truncated_normal([1, feature_size, NUM_FILTERS], stddev=1.0))
    b_feat = tf.Variable(tf.zeros(NUM_FILTERS))
  
    conv_feature = tf.nn.conv1d(
            tweet_f_input,
            W_feat,
            stride=1,
            padding='VALID')

    feat_adj = tf.nn.bias_add(conv_feature, b_feat)


    # Convolution
    pooled_outputs = []
    for i, filter_size in enumerate(FILTER_SIZES):
        
        emb_filter_shape = [filter_size, EMBEDDING_SIZE, 1, NUM_FILTERS]
        W_embedding = tf.Variable(tf.truncated_normal(emb_filter_shape, stddev=0.1))
        b_embedding = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]))

        conv_embedding = tf.nn.conv2d(
                embedded_expanded,
                W_embedding,
                strides=[1, 1, 1, 1],
                padding="VALID")


        conv_bias = tf.nn.bias_add(conv_embedding, b_embedding)

        conv_adjusted = conv_bias * feat_adj
        conv_relu = tf.nn.relu(conv_adjusted)

        # local pooling of convolution
        pooled = tf.nn.max_pool(
            conv_relu,
            ksize=[1, TWEET_WL_MAX - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding="VALID")

        pooled_outputs.append(pooled)

    num_filters_total = NUM_FILTERS * len(FILTER_SIZES)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    pool_weights = tf.Variable(
            tf.truncated_normal([num_filters_total, categories]))

    pool_biases = tf.Variable(
            tf.zeros([categories]))

    return (tweet_w_input, tweet_f_input, tf.nn.softmax(tf.nn.bias_add(tf.matmul(h_pool_flat, pool_weights), pool_biases)))
