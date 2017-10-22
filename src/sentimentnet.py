import tensorflow as tf

TWEET_WL_MAX = 70
EMBEDDING_SIZE = 5
FILTER_SIZES = [4] #, 4, 5]
#NUM_FILTERS = 15

JOIN_LAYER_SIZE = 4 

def create_graph(categories, vocab_size, feature_size):
    variables = []

    tweet_w_input = tf.placeholder(tf.int32, [None, TWEET_WL_MAX])
    tweet_f_input = tf.placeholder(tf.float32, [None, TWEET_WL_MAX, feature_size])
    # add dropout?

    # EMBEDDING
    W = tf.Variable(
            tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0))
    variables.append(W)
    embedded_chars = tf.nn.embedding_lookup(W, tweet_w_input)

    W_embed = tf.Variable(tf.truncated_normal([5, EMBEDDING_SIZE, JOIN_LAYER_SIZE], stddev=0.2))
    variables.append(W_embed)
    b_embed = tf.Variable(tf.truncated_normal([JOIN_LAYER_SIZE], stddev=0.2))
    variables.append(b_embed)

    embed_reshape = tf.nn.conv1d(
            embedded_chars,
            W_embed,
            stride=1,
            padding="VALID")
    embed_biased = tf.nn.bias_add(embed_reshape, b_embed)

    W_feat = tf.Variable(tf.truncated_normal([5, feature_size, JOIN_LAYER_SIZE], stddev=0.1))
    variables.append(W_feat)
    b_feat = tf.Variable(tf.truncated_normal([JOIN_LAYER_SIZE], stddev=0.1))
    variables.append(b_feat)
  
    feature_reshape = tf.nn.conv1d(
            tweet_f_input,
            W_feat,
            stride=1,
            padding="VALID")
    feature_biased = tf.nn.bias_add(feature_reshape, b_feat)

    joined = tf.concat([tf.expand_dims(embed_biased, -1), tf.expand_dims(feature_biased, -1)], 3)
   
    W_join = tf.Variable(tf.truncated_normal([1, 1, 2, 1]))
    variables.append(W_join)
    b_join = tf.Variable(tf.truncated_normal([1]))
    variables.append(b_join)

    conjoin = tf.nn.conv2d(
        joined,
        W_join,
        strides=[1, 1, 1, 1],
        padding="VALID")
    conjoin_bias = tf.nn.bias_add(conjoin, b_join)

    #pooled_outputs = []
    #for i, filter_size in enumerate(FILTER_SIZES):
        
    #    emb_filter_shape = [filter_size, JOIN_LAYER_SIZE, 1, NUM_FILTERS]
    #    W_embedding = tf.Variable(tf.truncated_normal(emb_filter_shape, stddev=0.1))
    #    variables.append(W_embedding)
    #    b_embedding = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]))
    #    variables.append(b_embedding)
#
#        conv_embedding = tf.nn.conv2d(
#                embedded_adjusted,
#                W_embedding,
#                strides=[1, 1, 1, 1],
#                padding="VALID")
#
#        conv_bias = tf.nn.bias_add(conv_embedding, b_embedding)
#        conv_relu = tf.nn.relu(conv_bias)
#
#        # local pooling of convolution
#        pooled = tf.nn.max_pool(
#            conv_relu,
#            ksize=[1, TWEET_WL_MAX - filter_size + 1, 1, 1],
#            strides=[1, 1, 1, 1],
#            padding="VALID")

#        pooled_outputs.append(pooled)

#    num_filters_total = NUM_FILTERS * len(FILTER_SIZES)
#    h_pool = tf.concat(pooled_outputs, 3)
#    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

#    pool_weights = tf.Variable(
#            tf.truncated_normal([num_filters_total, categories]))
#    variables.append(pool_weights)

#    pool_biases = tf.Variable(
#            tf.zeros([categories]))
#    variables.append(pool_biases)

    print(conjoin_bias.shape)

    end_width = (TWEET_WL_MAX - 5 + 1)* JOIN_LAYER_SIZE 

    reduxd = tf.reshape(conjoin_bias, [-1, end_width])

    W_final = tf.Variable(tf.truncated_normal([end_width, categories], stddev=1))
    variables.append(W_final)
    b_final = tf.Variable(tf.zeros([categories]))
    variables.append(b_final)

    final = tf.nn.bias_add(tf.matmul(reduxd, W_final), b_final)

    return (tweet_w_input, tweet_f_input, final, variables)

