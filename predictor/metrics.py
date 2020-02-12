import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def masked_f1(preds, labels, mask):
    """f1score with masking."""
    depth = tf.shape(preds)[-1]
    preds = tf.one_hot(tf.argmax(preds, 1), depth)
    # labels = tf.one_hot(tf.argmax(labels, 1), depth)

    mask = tf.cast(mask, dtype=tf.float32)

    precisions = [0, 0]
    recalls = [0, 0]
    f1s = [0, 0]

    y_true = tf.expand_dims(mask, 1) * tf.cast(labels, tf.float32)
    y_pred = tf.expand_dims(mask, 1) * tf.cast(preds, tf.float32)

    for i, axis in enumerate([None, 0]):
        TP = tf.cast(tf.count_nonzero(y_pred * y_true, axis=axis), tf.float32)
        FP = tf.cast(tf.count_nonzero(y_pred * (y_true - 1), axis=axis), tf.float32)
        FN = tf.cast(tf.count_nonzero((y_pred - 1) * y_true, axis=axis), tf.float32)

        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        precisions[i] = tf.reduce_mean(precision)
        recalls[i] = tf.reduce_mean(recall)
        f1s[i] = tf.reduce_mean(f1)

    micro_precision, macro_precision = precisions
    micro_recall, macro_recall = recalls
    micro_f1, macro_f1 = f1s
    return micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1