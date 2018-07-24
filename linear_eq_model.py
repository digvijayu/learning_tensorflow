import tensorflow as tf

# Model parameters
W = tf.Variable(.3, tf.float32);
b = tf.Variable(-.3, tf.float32);

# Input/Output parameters
x = tf.placeholder(tf.float32);

linear_model = W * x + b

y = tf.placeholder(tf.float32);

# Loss
squared_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_delta)

optimizer = tf.train.GradientDescentOptimizer(0.01);
train = optimizer.minimize(loss);

init = tf.global_variables_initializer()

session = tf.Session()

session.run(init);

for i in range (1000):
    session.run(train, {x : [1,2,3,4], y: [0, -1, -2, -3]})

# print(session.run(loss, {x : [1,2,3,4], y: [0, -1, -2, -3]}));

print(session.run([W, b]))

# session.close();
