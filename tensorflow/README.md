## Nine Steps To TensorFlow

NOTE:  These are the very minimal steps to get things going, they don't necessarily set you up to view this in TensorBoard.



1.  Define model parameters - `tf.Variable` objects - things you want TensorFlow to figure out for you - e.g., a weights matrix `W` and a bias vector `b`.
2.  Define model placeholders - `tf.placeholder` objects - things you'll be feeding in, e.g., X and Y values for training, and X values for validation and testing.
3.  Define your model - for example, a linear model of `y = W * x + b`.
4.  Define your loss function - for regression, you might use mean squared error (MSE), for categorization, it's considered best practices to use cross entropy.
5.  Define your optimizer - e.g., GradientDescentOptimizer for gradient descent.  It's **critical** to use a good learning rate, too small and your model will never converge, too large and your weights and biases will blow up.
6.  Tell the optimizer what to do, e.g., `optimizer.minimize(loss)` to minimize the loss function.  This is commonly called `train` in the code.
7.  Define your training data - when working with a Pandas DataFrame, turn your X values into a numpy array.  You may need to reshape your Y values to (-1, N), where N is the number of training cases.
8.  Create and run a TensorFlow Session() object, passing in the global variables initializer function.
9.  Setup training data as a dictionary with keys of `x` and `y`, fed to another call to session run as the `feed_dict` parameter.  Anything you pass as the `fetches` argument in `tf.run()` will be available for you to use outside of the loop, e.g., to monitor training loss.
