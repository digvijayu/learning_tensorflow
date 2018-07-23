import tensorflow as tf

const1 = tf.constant(5.0)
const2 = tf.constant(6.0)

mult = const1 * const2

session = tf.Session()

File_Write = tf.summary.FileWriter(
    'session_graph',
    session.graph
    )

print(session.run(mult))

session.close()
