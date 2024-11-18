import tensorflow as tf


def loss_func(pred, y, loss, datasets):
    if loss == 0:  # weighted softmax
        ratio = datasets.train.pos_instance_ratio()
        classes_weights = tf.constant([[1.0 - ratio, ratio]])
        weight_per_label = tf.transpose(tf.matmul(y, tf.transpose(classes_weights)))
        xent = tf.multiply(weight_per_label, tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,
                                                                                        labels=y))  # shape [1, batch_size]
        cost = tf.reduce_mean(xent)
    elif loss == 1:  # softmax
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

    elif False:  # optimized pairwise exponential function without grouping, only for reference
        pred_scaled = tf.nn.softmax(pred)
        pred_1 = tf.slice(pred_scaled, [0, 0], [-1, 1])
        y_1 = tf.slice(y, [0, 0], [-1, 1])
        positive = tf.greater(y_1, 0.0)
        negative = tf.less(y_1, 1.0)
        pred_x = tf.boolean_mask(pred_1, positive)
        pred_x = tf.reshape(pred_x, [-1, 1])
        pred_y = tf.boolean_mask(pred_1, negative)
        pred_y = tf.reshape(pred_y, [1, -1])
        diff = tf.subtract(pred_x, pred_y)
        diff_exp = tf.exp(-diff)
        cost = tf.reduce_sum(diff_exp)

    return cost
