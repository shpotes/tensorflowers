import tensorflow as tf

def evaluate(target, logits):
    metric = tf.keras.metrics.SparseCategoricalCrossentropy(
        from_logits=True
    )

    logits /= tf.reduce_sum(logits, axis=-1)

    metric.update_state(
        target, logits
    )

    return metric.result()