


Note:  These are mostly taken from the [Python API Documentation](https://www.tensorflow.org/api_docs/python/)


| Object                          | Notes                                     |
|---------------------------------|-----------------------------------------------------------| 

| `tf.Variable`    | What you want TF to figure out, e.g., weights and biases |
| `tf.constant`    | Immutable, per the name |
| `tf.Placeholder` | What you'll be feeding into the model, e.g., X and Y, can leave the first size parameter as `None` if you may change the batch size with each iteration |
| `tf.Session` | A class for running TensorFlow operations - by convention called `sess` - `sess.run()` allows you to evaluate values in `tf.Variable` and `tf.placeholder` objects |
| `tf.matmul()` | Operation to do matrix multiplication |
| `tf.nn` | Neural network goodies in here |
| `tf.nn.sigmoid` | Activation function in S-shape, main issue is that at extreme values, slope is zero, so this can hose things up when computing gradients |
| `tf.nn.relu` | Rectified linear unit - activation function that has taken over in popularity from sigmoid |
| `tf.random_normal()` | Outputs random values from a normal distribution - often used to initialize weights in a matrix |
| `tf.zeros()` | Creates a tensor with all elements set to zero - OK to use for initializing biases (?) but can be very problematic when used to initialize weights |