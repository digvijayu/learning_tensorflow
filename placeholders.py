import tensorflow as tf

placeholder1 = tf.placeholder(tf.float32)
placeholder2 = tf.placeholder(tf.float32)

add_node = placeholder1 + placeholder2

session = tf.Session();

output = session.run(add_node, {placeholder1: [1,2],placeholder2: [3, 4]});

print(output);

session.close();
