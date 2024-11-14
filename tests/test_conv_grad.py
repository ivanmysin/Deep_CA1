import tensorflow as tf

a = tf.Variable([1, 2, 3], dtype=tf.float32)
a = tf.reshape(a, shape=(3, 1, 1))

b = tf.Variable([-1, 0, 1], dtype=tf.float32)
b = tf.reshape(b, shape=(1, 3, 1))

with tf.GradientTape() as tape:
    c = tf.nn.conv1d(b, a, stride=1, padding='SAME')

    cs = tf.math.reduce_sum(c)
    gradients = tape.gradient(cs, a)

print(cs)
print("gradients: ", gradients)
