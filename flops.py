import tensorflow as tf

tf.reset_default_graph()


tf.train.import_meta_graph('-27962.meta')
saver = tf.train.Saver()
with tf.Session() as sess:
    # The session is binding to the default global graph
    tf.profiler.profile(
        sess.graph,
        options=tf.profiler.ProfileOptionBuilder.float_operation())

    parameters = tf.profiler.profile(
        sess.graph,
        options=tf.profiler.ProfileOptionBuilder
        .trainable_variables_parameter())
    print ('total parameters: {}'.format(parameters.total_parameters))
