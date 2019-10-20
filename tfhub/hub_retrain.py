import tensorflow as tf
import tensorflow_hub as hub

num_classes = 10
m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4", output_shape=[2048],
                   trainable=False),  # Can be True, see below.
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
m.build([None, 299, 299, 3])  # Batch input shape.
m.summary()
