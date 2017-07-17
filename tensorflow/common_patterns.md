

### Common Patterns

<pre>
import tensorflow as tf

# setup placeholders for your X and Y_ values you'll feed to the training function

# we're dealing with 28x28 grayscale images here, with the pixels flattened, thus the 784, 1
# we're using None to indicate we may feed in different batch sizes

X = tf.placeholder(tf.int32, [None, 784, 1])

# 10 possible categories for the image, digit 0-9
# again, we're using None to indicate a variable number of cases

Y_ = tf.placeholder(tf.int32, [None, 10])

# set the learning rate to something small (here, I used 0.003, which didn't blow up the weights)
learn_rate = 0.003

# create an optimizer
optimizer = tf.train.GradientDescentOptimizer(learn_rate)

# define a training step - this is what will be evaluated during each step of the training
train_step = optimizer.minimize(cross_entropy)

# create a session object
sess = tf.Session()

# initialize all variables
sess.run(init)

# set the number of training iterations and the batch size (number of cases to feed to the model on each training iteration)
training_steps = 10000
batch_size = 100


for i in range(training_steps):
    # load batch of images and labels
    batch_X, batch_Y = dataset.train.next_batch(batch_size)
    train_data = {X: batch_X, Y_: batch_Y}

    # train (here, we’re running a TensorFlow computation, feeding placeholders
    sess.run(train_step, feed_dict=train_data)

</pre>
