from __future__ import print_function

import cleanlab
import numpy as np
from util import input_utils, loss_utils as ut
import tensorflow as tf
from util import cl_tool

L2_value = 0.0001
learning_rate = 0.001
training_epochs = 55
batch_size = 500
dropout_rate = 0.7
display_step = 5
dump_step = 10


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale=L2_value)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
    return new_variables


def single_fc_layer(input_layer, input_dimension, output_dimension, keep_prob):
    weight = create_variables("weight", [input_dimension, output_dimension])
    bias = tf.Variable(tf.random_normal([output_dimension]))
    output_layer = tf.add(tf.matmul(input_layer, weight), bias)
    output_layer = tf.nn.dropout(output_layer, keep_prob)
    output_layer = tf.nn.relu(output_layer)
    return output_layer


def mutation_spec_first(spec, m1, m2, m3, complexity, similarity, keep_prob):
    model_size_times = 2
    with tf.variable_scope('seperate_mut', reuse=False):
        with tf.variable_scope('mut1', reuse=False):
            mut_1 = single_fc_layer(m1, 35, 35 * model_size_times, keep_prob)
        with tf.variable_scope('mut2', reuse=False):
            mut_2 = single_fc_layer(m2, 35, 35 * model_size_times, keep_prob)
        with tf.variable_scope('mut3', reuse=False):
            mut_3 = single_fc_layer(m3, 35, 35 * model_size_times, keep_prob)
        mut_concat = tf.concat([mut_1, mut_2, mut_3], 1)
        with tf.variable_scope('mut_concat', reuse=False):
            mut_concat = single_fc_layer(mut_concat, 35 * 3 * model_size_times, 35 * model_size_times, keep_prob)

    with tf.variable_scope('seperate_spec', reuse=False):
        with tf.variable_scope('spec', reuse=False):
            spec_1 = single_fc_layer(spec, 34, 34 * model_size_times, keep_prob)
        spec_concat = tf.concat([spec_1, mut_concat], 1)
        with tf.variable_scope('fc1', reuse=False):
            spec_concat = single_fc_layer(spec_concat, 69 * model_size_times, 32 * model_size_times, keep_prob)
    with tf.variable_scope('fc', reuse=False):
        with tf.variable_scope('complex', reuse=False):
            complex_1 = single_fc_layer(complexity, 21, 21 * model_size_times, keep_prob)
        with tf.variable_scope('similar', reuse=False):
            similar_1 = single_fc_layer(similarity, 15, 15 * model_size_times, keep_prob)
        fc_1 = tf.concat([spec_concat, complex_1, similar_1], 1)
        with tf.variable_scope('fc1', reuse=False):
            fc_2 = single_fc_layer(fc_1, 68 * model_size_times, 128, keep_prob)
    final_weight = create_variables("final_weight", [128, 2])
    final_bias = tf.get_variable("final_bias", shape=[2], initializer=tf.zeros_initializer())
    out_layer = tf.add(tf.matmul(fc_2, final_weight), final_bias)
    return out_layer


def run(trainFile, trainLabelFile, testFile, testLabelFile, loss, filepath, threshold):
    tf.reset_default_graph()
    # Network Parameters
    n_classes = 2  # total output classes (0 or 1)

    # tf Graph input
    x = tf.placeholder("float", [None, 175])
    spec = tf.placeholder("float", [None, 34])
    mutation1 = tf.placeholder("float", [None, 35])
    mutation2 = tf.placeholder("float", [None, 35])
    mutation3 = tf.placeholder("float", [None, 35])
    complexity = tf.placeholder("float", [None, 21])
    similarity = tf.placeholder("float", [None, 15])
    y = tf.placeholder("float", [None, n_classes])

    # dropout parameter
    keep_prob = tf.placeholder(tf.float32)

    # Construct model
    pred = mutation_spec_first(spec, mutation1, mutation2, mutation3, complexity, similarity, keep_prob)
    datasets = input_utils.read_data_sets(trainFile, trainLabelFile, testFile, testLabelFile)

    # Define loss and optimizer
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    y = tf.stop_gradient(y)
    cost = ut.loss_func(pred, y, loss, datasets)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost + regu_losses)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(datasets.train.num_instances / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = datasets.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)

                _, c, regu_loss = sess.run([optimizer, cost, regu_losses], feed_dict={spec: batch_x[:, :34],
                                                                                mutation1: batch_x[:, 34:69],
                                                                                mutation2: batch_x[:, 69:104],
                                                                                mutation3: batch_x[:, 104:139],
                                                                                complexity: batch_x[:, 139:160],
                                                                                similarity: batch_x[:, -15:],
                                                                                y: batch_y,
                                                                                keep_prob: dropout_rate})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step

            if epoch % display_step == 0:
                print("Epoch:", '%02d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost), ", l2 loss= ",
                        np.sum(regu_loss))

        probabilities = sess.run(tf.nn.softmax(pred), feed_dict={spec: datasets.train.instances[:, :34],
                                            mutation1: datasets.train.instances[:, 34:69],
                                            mutation2: datasets.train.instances[:, 69:104],
                                            mutation3: datasets.train.instances[:, 104:139],
                                            complexity: datasets.train.instances[:, 139:160],
                                                similarity: datasets.train.instances[:, -15:],
                                                y: datasets.train.labels,
                                                keep_prob: dropout_rate})

    # 识别可能的标签问题
    labels = datasets.train.labels[:, 0]

    # train_instances, train_labels = cl_tool.remove_noise_ori(labels, probabilities, ratio, datasets.train.instances,
    #                                                          datasets.train.labels)

    train_instances, train_labels = cl_tool.remove_noise_threshold(labels, probabilities, datasets.train.instances,
                                                                 datasets.train.labels, threshold)

    clean_datasets = input_utils.create_train_sets(train_instances, train_labels)
    print(clean_datasets.train.instances.shape)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(clean_datasets.train.num_instances / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = clean_datasets.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)

                _, c, regu_loss = sess.run([optimizer, cost, regu_losses], feed_dict={spec: batch_x[:, :34],
                                                                                mutation1: batch_x[:, 34:69],
                                                                                mutation2: batch_x[:, 69:104],
                                                                                mutation3: batch_x[:, 104:139],
                                                                                complexity: batch_x[:, 139:160],
                                                                                similarity: batch_x[:, -15:],
                                                                                y: batch_y,
                                                                                keep_prob: dropout_rate})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step

            if epoch % display_step == 0:
                print("Clean Epoch:", '%02d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost), ", l2 loss= ",
                        np.sum(regu_loss))


        # Write Result
        res = sess.run(tf.nn.softmax(pred), feed_dict={spec: datasets.test.instances[:, :34],
                                                mutation1: datasets.test.instances[:, 34:69],
                                                mutation2: datasets.test.instances[:, 69:104],
                                                mutation3: datasets.test.instances[:, 104:139],
                                                complexity: datasets.test.instances[:, 139:160],
                                                similarity: datasets.test.instances[:, -15:],
                                                y: datasets.test.labels,
                                                keep_prob: 1.0})
        with open('./Result/' + filepath, 'w') as f:
            for susp in res[:, 0]:
                f.write(str(susp) + '\n')
        # print(" Optimization Finished!")