


Note:  These are mostly taken from the [Python API Documentation](https://www.tensorflow.org/api_docs/python/)


| Object                          | Notes                                     |
|---------------------------------|-----------------------------------------------------------| 
| `tf.Variable`    | What you want TF to figure out, e.g., weights and biases |
| `tf.constant`    | Immutable, per the name |
| `tf.placeholder` | What you'll be feeding into the model, e.g., X and Y, can leave the first size parameter as `None` if you may change the batch size with each iteration |
| `tf.Session` | A class for running TensorFlow operations - by convention called `sess` - `sess.run()` allows you to evaluate values in `tf.Variable` and `tf.placeholder` objects |
| `tf.matmul()` | Operation to do matrix multiplication |
| `tf.nn` | Neural network goodies in here |
| `tf.nn.sigmoid` | Activation function in S-shape, main issue is that at extreme values, slope is zero, so this can hose things up when computing gradients |
| `tf.nn.relu` | Rectified linear unit - activation function that has taken over in popularity from sigmoid |
| `tf.nn.dropout(X, p)` | Given a layer and a percentage chance of dropout, sets weights and biases to neurons in that layer to zero with probability (1 - p). |
| `tf.random_normal()` | Outputs random values from a normal distribution - often used to initialize weights in a matrix |
| `tf.zeros()` | Creates a tensor with all elements set to zero - OK to use for initializing biases (?) but can be very problematic when used to initialize weights |
| `tf.add()` | Operation adding two nodes together |
| `tf.train.GradientDescentOptimizer` | Optimizer that implements the gradient descent algorithm | 
| `tf.layers` | provides a set of high-level neural networks layers |
| `tf.summary` | Tensor summaries for exporting information about a model |
| `tf.reshape()` | Reshape a tensor, e.g., for input.  pass -1 as the first value in the shape list parameter to retain same number of samples |
| `tf.equal()` | Tests x == y element-wise |
| `tf.argmax()` | Returns the index with the largest value across axes of a tensor - in ties, behavior is not guaranteed.  Works well for classification problems where the result is a series of percentages |


#### Resources

* [Python API Guides List](https://www.tensorflow.org/versions/master/api_guides/python/)
* [Constants, Sequences, and Random Values](https://www.tensorflow.org/versions/master/api_guides/python/constant_op)
* [Training and Optimizers](https://www.tensorflow.org/api_guides/python/train)
