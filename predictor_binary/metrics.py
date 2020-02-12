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
    tp = tf.multiply(tf.cast(tf.equal(tf.argmax(preds, 1), 1), tf.float32), 
                     tf.cast(tf.equal(tf.argmax(labels, 1), 1), tf.float32)) #tp 1
    fp = tf.multiply(tf.cast(tf.equal(tf.argmax(preds, 1), 1), tf.float32), 
                     tf.cast(tf.equal(tf.argmax(labels, 1), 0), tf.float32)) #fp 1
    fn = tf.multiply(tf.cast(tf.equal(tf.argmax(preds, 1), 0), tf.float32), 
                     tf.cast(tf.equal(tf.argmax(labels, 1), 1), tf.float32)) #fn 3

    mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    tp *= mask
    tp = tf.reduce_mean(tp)
    fp *= mask
    fp = tf.reduce_mean(fp)
    fn *= mask
    fn = tf.reduce_mean(fn)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    return precision, recall, f1_score
