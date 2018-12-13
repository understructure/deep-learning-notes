


```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git

gcloud compute zones list

datalab create mydatalabvm --zone us-east1-b
```

tf.enable_eager_execution()

`model.evaluate()` - use this with the validation set to see how good your model is.

The Python API layer treats TensorFlow as a numeric processing library

Estimators

tf.estimator lets you wrap your own model you would build from layers using the tf.layers API
but of course no need to write your own model most of the time, just use TF's built-in models

DNNLinearCombinedClassifier - "work horse of Enterprise Machine Learning"

[tf.feature_column](https://www.tensorflow.org/api_docs/python/tf/feature_column)

Training:   Create a function that returns `features, labels` and pass this function as the first argument of `model.train()`

Predicting: Create a function that returns `features` and pass this function as the first argument of `model.predict()`

1.  Create a list of `tf.feature_column` objects
2.  Create a `model` object with an estimator, pass in list of feature columns and appropriate parameters / values for that estimator
3.  Call model.train() and pass in the function you created to create the training data set and the number of steps (if desired)
4.  Call model.predict and pass in the function you created to create the features for the test set.  Assign this to a variable `predictions`, which becomes a generator.
5.  call `next()` on the `predictions` object you created in the previous step to see each prediction.


#### Checkpoints

* If you want to save your checkpoints, just pass a folder name in your `tf.estimator` call.  
* You can load a trained model in this way.
* If you continue training, it picks up from where it last left off if the specified directory exists.
* Delete checkpoints directory to restart training (especially when you've changed the model).


#### Training on in-memory data sets

* If your data fits in memory, can use one of the following:
	* `tf.estimator.inputs.numpy_input_fn` - return a dict of np.arrays for features as `x`, np.array of labels as `y`
	* `tf.estimator.inputs.pandas_input_fn` - return the df as `x`, `df['target_column']` as `y`
* `steps` parameter - runs that number of minibatches through from last checkpoint (if it exists)
* `max_steps` parameter - if checkpoint has already hit this, will do nothing

#### Train on large datasets with Dataset API

* Input functions are called *only once* when your model is instantiated (not every time you need new data)
* Use `tf.data.Dataset` class and one of:
	* `.TextLineDataset`
	* `.TFRecordDataset`
	* `.FixedLengthRecordDataset`
* Use `map()` and a function for each line of a dataset
* The contract for an input function is to return one TensorFlow node representing the features and labels expected by the model (during training or inference)
* This node will be connected to the inputs of the model
* Delivers a fresh batch of data every time it's executed
* Input functions don't return data, the nodes of the graph return data each time they're executed
* Dataset API delivers batch of data at each training step
* Also makes sure data is loaded progressively and never saturates the memory
* `dataset.make_one_shot_iterator().get_next()` gets a TensorFlow node that, each time it gets executed during training returns a batch of training data
* Working with multiple similarly-named files:
	* can use `list_files` parameter in `tf.data.Dataset` and use glob-like patterns (e.g., asterisk)
	* `flat_map` does one to many transformations - maps all the lines in the file into a dataset
	* `map` does one to one transformations - applies a function to each line, e.g., to parse CSV lines


#### Big jobs, Distributed training

* Use `tf.estimator.train_and_evaluate()` - need to do four things:

* Choose an estimator
* Provide a run configuration
* Provide training data through `TrainSpec` - have to use `dataset` API to set it up properly
* Provide evaluation data through `EvalSpec` - have to use `dataset` API to set it up properly
* Keep in mind, in distributed TensorFlow, evaluation happens on a dedicated server, so for `EvalSpec` you'll only get evaluations only as often as you entered for your model checkpointing in your config
* Use the `throttle_secs` parameter in the `EvalSpec` to get these *less* frequently than every time it checkpoints
* `run_config = tf.estimator.RunConfig(model_dir=output_dir, save_summary_steps=100, save_checkpoints_steps=2000)`
* `estimator = tf.estimator.LinearRegressor(config=run_config)`
* `train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=50000)`
* `eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100, throttle_secs=600, exporters=...)`
* `tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)`
* NOTE:  SGD really *needs* well-shuffled data to work correctly, and if you don't use `dataset.shuffle()` in distributed TF, every worker will see the exact same records adn will produce the exact same gradients, so parallelization is no benefit.

#### Monitoring with TensorBoard

* Look at the graph called "fraction of zero values" - if all neurons output zeros, your NN is dead
* `tf.summary.text()` and `tf.summary.scalar()` will summarize variables for TensorBoard, also image, audio, and histogram in `tf.summary`

#### Serving Input Function

* `gcloud ml-engine local predict` command line can run predictions for you without even deploying your model
* Testing JSON files are one JSON object per file line
* You'll create a serving input function to transform incoming JSON into what your model expects for the features
* This is done automatically?? - To decompress images into base 64, feature name must end in `_bytes` and must have a dict / key-value pair of `b64` and base 64-encoded string as its value
* `tf.estimator.LatestExporter` - just uses the most recent model version in your directory - they're all in there in subdirectories named with a UNIX-style timestamp (e.g., seconds since 1/1/1970)


#### Train A Model

* Split code into task.py file and model.py file
* task.py - parses command line parameters and sends to `train_and_evaluate()` function, invokes model.py
* model.py - fetches data, defines features, configures service signature, runs actual train and eval loop
* TF & Python require this package structure:

```
dir_name/
dir_name/PKG-INFO
dir_name/setup.cfg
dir_name/setup.py
dir_name/trainer/
dir_name/trainer/__init__.py
dir_name/trainer/task.py
dir_name/trainer/model.py
```

* Try calling this locally with something like `python -m trainer.task`
* Be sure to select a single-region bucket for ML

#### Monitoring and Deploying Training Jobs

* `gcloud ml-engine jobs describe job-name`
* `gcloud ml-engine jobs stream-jobs job-name`


#### Statistics vs. ML

* Statistics - this is all the data I'm ever going to get, make the most of it, impute values, remove outliers
* ML - I can get tons of data, so keep everything, create separate models for outliers, create extra columns for missing values - "is_missing"


#### Preprocessing and Feature Creation


* Scaling features b/t 0 and 1 helps algorithms like gradient descent
* `tf.feature_column.categorical_column_with_vocabulary_list` method - one hot encoding
* `tf.feature_column.bucketized_column` method - turn real-valued feature into discrete
* Apache Beam - can use for aggregations like number of sales in previous hour
* Cloud DataFlow
* Beware using SQL for preprocessing - have to always make sure that the same transformations happen during training, test, and validation


#### Beam and Dataflow

* Cloud Dataflow - think in terms of pipelines - change data from one format to another
* Can run programs written in Java and Python
* Serverless, fully-managed - changes amount of resources elastically
* Use Apache Beam API, deploy to Cloud Dataflow
* Beam can be executed on Flink, Spark, etc.
* Beam supports both batch and streaming data using same pipeline code
* Beam = Batch + Stream
* `beam.io` - input output
* Beam has a variety of connectors to use services on GC like BigQuery
* Can write your own connectors
* Each step in Beam is a transform
* Every step gets a PCollection and outputs a PCollection
* Last step of a pipeline outputs to a "sink"
* To run a pipeline, you need a "runner" - executes pipeline code
* Runners are platform specific
	* Cloud Dataflow runner
	* Spark runner
	* Direct runner - executes pipeline on local machine

```import apache_beam as beam
p = beam.Pipeline(argv=sys.argv)
```

* In Python, the pipe operator is overloaded to call the "apply" method - just chain these together
* `p.run()` to run the pipeline
* PCollection doesn't store all data in memory - it's like a data structure with pointers to where the data flow cluster stores your data
* `beam.io.WriteToText` shards output by default to avoid file contention issues
* To execute a Pipeline on GCP, you need to pass:
	* Name of project
	* Cloud storage bucket for staging
	* Cloud storage bucket for temp data
	* Name of the runner (e.g., `DataFlowRunner`)

#### Data Pipelines that Scale

* A ParDo acts on one item at a time (like a Map in MapReduce) - stateless.  Useful for:
	* Filtering
	* Extracting parts of an input
	* Converting one Java type to another
	* Calculating values from different parts of inputs
* `beam.Map` - 1:1 transforms
* `beam.FlatMap` - any number of outputs for a given input, including zero
* `total_amount = sales_amounts | Combine.globally(sum)` - creates overall sum of sales_amounts values
* `total_sales_per_person = sales_records | Combine.perKey(sum)` - grouped key-value pair


#### Memorization vs. Generalization

* Feature Crossing - think of blue in quadrants I and III, yellow in quadrants I and IV - multiply x-axis and y-axis to get positive or negative, that makes this a linear problem
* However, feature crossing is essentially memorization, not generalization - should you do this?
* Memorization works when you have so much data that for any single grid cell in your input space, the distribution of data is statistically significant.
	* You're essentially just learning the mean for every grid cell
* Neural networks provide another way to learn highly complex spaces. But feature crosses let linear models stay in the game. 
	* Without feature crosses, the expressivity of linear models would be quite limited. 
	* With feature crosses, once you have a massive dataset, a linear model can learn the nooks and crannies of your input space
* NNs with many layers are non-convex
* Optimizing linear problems is a convex problem
* Convex problems are WAY easier than non-convex problems


#### Sparsity + Quiz

* Internally, TensorFlow uses sparse encoding for both one-hot encodings and feature crosses, very efficient
* Feature crosses lead to sparsity


#### Too Much Of A Good Thing

* Sometimes all you need are the raw inputs
* If you do a bunch of unnecessary feature engineering, you can overfit
* Can somewhat test for this on test data set
* L1 regularisation actually removes features


#### Implementing Feature Crosses

* `day_hour = tf.feature_column.crossed_column([day_of_week, hour_of_day], 24 * 7)`
* Above, 24 * 7 = number of hash buckets
* The number of `hash_buckets` controls sparsity and collisions
* Small `hash_buckets` -> lots of collisions
* Large `hash_buckets` -> very sparse
* In practice, dude uses a number somewhere between `1/2 * sqrt(n)` and `2n` as his own rule of thumb


#### Embedding Feature Crosses

* Pass sparse representation to an embedding layer - esentially one-hot encoded
* Weights for each embedding are learned
* By passing it to hidden (?) nodes that have real-valued values, (e.g., two nodes) these two values now represent one of the 168 day/hour combinations
* As in NLP, if two day/hour combinations are similar, their real-valued representations will also be similar
* By this, we say the model learns to embed the feature cross in a lower-dimensional space
* If these were learned for, say, traffic in London, you could reuse the embeddings when setting up a new model for say, Frankfurt
* Just load saved model and tell it not to train that layer


#### Where to Do Feature Engineering

* Your model consists of:
	* Input function to read the data
	* Feature columns to act as placeholders for the things you read
	* An estimator you create passing in the feature columns
	* train spec, eval spec, exporter, etc. and call train and evaluate
*  Three possible places:

1. On the fly as you read in data (in input function itself or by creating feature columns)
2. As a separate step before training (do in data flow so you can do this at scale in a distributed manner)
3. Dataflow + TensorFlow (`tf.transform`) - do preprocessing in data flow, create a set of preprocessed features, but tell prediction graph you want the same transformations carried out in TF during serving

* Bucketizing - for numeric values (e.g., latitude and longitude) - can use `np.linspace()` to split into n buckets, then use `bucketized_column()` and/or `crossed_column()` with that


#### Feature Creation in TensorFlow

* Create a function (like `add_engineered()`) to add features, call it from all the input functions (training, evaluation, serving)
* In `serving_input_function()`, do something like `return ServingInputReceiver(add_engineered(features), json_features_placeholder)`

#### Feature Creation in DataFlow

* Using Dataflow for preprocessing works best if it's also part of your prediction runtime
* Dataflow is ideal for features that require time-windowed aggregations
* Adding new features in Dataflow is like any other `PTransform` (remember, it's Apache Beam)
* Dataflow is great for working with all the data
* TensorFlow is great for adding features on the fly

#### TensorFlow Transform


* TF models tend to be stateless
* `tf.transform` - do feature transformations at scale, and on streaming data too
	* limited to TF methods, but you get all the efficiency of TensorFlow and stuff like aggregates too
* `tf.transform` uses Dataflow during training, but only TensorFlow during prediction
* `tf.transform` - hybrid of Apache Beam and TensorFlow
* When you hear Dataflow, think back end preprocessing for ML models
* For on-the-fly processing for ML models, think TensorFlow
* `AnalyzeAndTransformDataset` - executed in Beam to create training dataset
* `TransformDataset` - executed in Beam to create the evaluation dataset
* The underlying transformations are executed in TensorFlow at prediction time


#### Regularization

* Minimize training error, but balance against complexity.  Don't overfit, but also a model that's too simplified is useless.
* L1 regularization
* L2 regularization
* Max-norm regularization

#### L1 & L2 Regularizations

* L1 and L2 represent model complexity as the magnitude of the weight vector, and try to keep that in check
* Linear algebra - magnitude of a vector is represented by the norm function (e.g., L1 and L2 norm)
* L2 norm - aka "weight decay" the most common, can be represented without subscript, like ||w|| - take square root of sum of squared values
* L1 norm - Sum of absolute values
* If we keep the magnitude of our weight vector smaller than some number, we've achieved our goal
* In L2 regularization, the complexity of the model is defined by the L2 norm of the weight vector
	* L(w, D) + lambda ||w|| (sub 2 for L2 norm)
		* L(w, D) - minimize training error
		* ||w|| (sub 2 for L2 norm) - but balance against complexity
		* lambda - controls how these are balanced
* In ML, we cheat a bit by squaring the L2 norm to make calculation of derivatives easier
* To do L1 regularization, just substitute L1 norm for L2 norm, but be careful because this results in a different solution, one that's much more sparse
* Sparsity here refers to the fact that some of the weights will end up having optimal value of zero
	* Therefore, L1 regularization can be used as a feature selection mechanism
* Complex models are bad.  One way to keep them simple is to apply regularization and adjust lambda (regularization rate) until we achieve acceptable performance

#### Learning rate and batch size

* Model performance is very sensitive to learning rate and batch size
* TensorFlow's `LinearRegressor` has a default learning rate of 0.2 or 1/sqrt(num_features), whichever is smaller
* If batch size is too small, it might not be a good representation of the input
* If batch size is too large, training will take a very long time
* 40-100 tends to be a good size for batch size, but can go up to 500
* Shuffling - by shuffling the dataset, you'll help make sure that each batch is representative of the entire training set, and that the loss is approximately the same for each batch

#### Optimization

* Minimizing / maximizing some function of f(x) by altering x
* Gradient Descent - Traditional approach, typically implemented stochastically (w/ batches)
* Momentum - Reduces learning rate when gradient values are small
* AdaGrad - Gives frequently-occurring features lower learning rates
* AdaDelta - Improves AdaGrad by avoiding reducing learning rate to zero
* Adam - Basically AdaGrad with a bunch of fixes
* Ftrl - "Follow the regularized leader" - works well on wide models
* Adam and Ftrl make good defaults for DNN and linear models

#### Practicing with Tensorflow code

* Batch size - in your input function, e.g., `pandas_input_fn()`
* Learning rate - in your optimizer
* L2 regularizaton amount - in your optimizer
* Pass optimizer to the estimator object
* Use `steps` rather than `epochs` in `model.train()` - epochs isn't failure friendly in distributed training
* `steps = epochs * n_examples / batch_size`
* `num_steps = (len(traindf) / batch_size) / learning_rate`
* if you decrease learning rate, you'll have to train for more epochs


#### Lab - some rules of thumb in tuning models

* Training error should steadily decrease, steeply at first, and then plateau
* If training error doesn't converge, try running it longer
* If training error decreases too slowly, increasing learning rate may help it decrease faster
	* But sometimes exact opposite can happen if learning rate is too high
* If training error varies widely, try decreasing learning rate
	* Lower LR + larger number of steps or larger batch size is often a good combination
* Very small batch sizes can cause instability.  First try larger values (e.g., 100, 1000) and decrease until you see degredation
* Don't take these as gospel, experiment!

* If your batch size is too high, your loss function will converge slowly
* If your batch size is too low, your loss function will oscillate wildly



* Check out the visualization for [loss function](http://cs231n.github.io/neural-networks-3/)
* [PDF whitepaper on Google Vizier](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)

#### Regularization for sparsity

* Remember that L1 can be used to help select features - any 0's represent features that can be dropped
* Fewer coefficients to load means less memory and a smaller model
* Fewer multiplications needed means prediction speed is faster (training speed too)
* Optimizing for L1 regularization is a NP-hard, non-convex optimization problem
* L1 and L2 plotted - L1 is pointy at zero (diamond), L2 is rounder at zero (circle)
* L1 normalization will take two similar features and throw one away
* L2 normalization will keep two similar features and keep their weight magnitudes small
* L1 means smaller model that may be less predictive
* Elastic nets - best of both worlds - combine feature selection of L1 with generalizability of L2
	* Two different lambdas to tune now though
* L2 regularization penalizes large weight values more

#### Logistic Regression

* Logistic regression - transform linear regression by a sigmoid activation function
* Often we regularization with early stopping to counteract overfitting
* Early stopping is an approximate equivalent of L2 regularization, often used in its place because it's computationally cheaper
* In practice, we use L1, L2 and early stopping
* Use a threshold, e.g., 50% or higher is a yes
* TP, TN, FP, FN
* Precision - TP / all positives
* Recall / sensitivity / TP rate - TP / (TP + FN) (or anything you **predicted** was true)
* Tune threshold to optimize metric of your choice
* Use the ROC curve to choose the decision threshold based on decision criteria
	* ROC is built by looking at FP rate (x-axis) by TP rate (y-axis) for all possible threshold choices
* Use AUC (area under curve) as an aggregate measure of performance across all possible classification thresholds

#### Training NNs

* Optimizers
	* `tf.train.AdamOptimizer()`
		* learning_rate
* Estimators
	* `tf.estimator.DNNRegressor()`
		* hidden_units
		* feature_columns
		* optimizer
		* dropout
* Vanishing gradients - e.g., when using sigmoid or tanh activations in hidden layers, slope gets closer to zero, backprop - gradient gets smaller and smaller - compounding all these small gradients until it vanishes
	* To fix, use ReLUs, ELUs, and other non-linear activation functions
* Exploding gradients - learning rates are important here
	* To fix - batch normalization, gradient clipping, weight regularization, smaller batch sizes, batch normalization
	* Batch normalization - get mini batch's mean and standard deviation, normalize inputs to that node, scale and shift by gamma * x + beta (gamma and beta are learned parameters)
* IDeally, keep gradients as close to 1 as possible, especially for very deep nets, so you don't compound and eventually overflow or underflow
* ReLU layers can die - montior fraction of zero weights in TensorBoard (they quit working when activations are zero due to values being in negative domain)
	* Solution - lower your learning rates
* Linear scaling - scale b/t 0 and 1
* Hard capping / clipping - scale b/t -1 and 1
* Log scaling - apply log - great when data has a huge range
* Standardization - essentially z-scores
* Remember, when using dropout, only use in training!
	* Activations scaled over (1/(1 - drop probability)) or inverse of keep probability
	* Effectively creates an ensemble model - each pass, batch sees a different network
	* dropout usually b/t 10-50% (dropout 0.2 is typical)
	* Best to use on larger networks
	* more you dropout, stronger the regularization
* `np.linspace()` - useful for bucketizing numeric range columns
* `mask = np.random.rand(len(df)) < 0.8` - make a DF of true and false values of same length as , 80% of which will be true
* If bucketing lat and long for California, keep in mind CA is longer than it is wide, they bucketed into 5 and 10 values per bucket
* If you scale predicted value going into the model, make sure to unscale it after you get your predicted value


#### Multi-class Neural Networks

* "one vs. one method" - you can create binary combinations for each combination of classes, but you'd get (n * (n - 1)) / 2 combinations, so a four-class problem would produce (4 * 3) / 2 = 6 models
	* this is inefficient and doesn't scale
* "one vs. all" or "one vs. rest" - output a vector with probabilties for each class
* Softmax - `tf.nn.softmax_cross_entropy_with_logits_v2()`
* For multiclass multilabel outputs - `tf.nn.sigmoid_cross_entropy_with_logits()`
* Approximate versions of Softmax exist - generally used during training but full Softmax used during inference for better accuracy
* To do this, change default partition strategy from `mod` to `div`
	* Candidate sampling - calculates for all the positive labels, but only for a random sample of negatives - `tf.nn.sampled_softmax_loss()`
		* Number of negatives is important hyperparameter
		* More intuitive - doesn't require a really good model
	* Noise-contrastive - approximates the denominator of Softmax by modeling the distribution of outputs - `tf.nn.nce_loss()`
		* Denominator of Softmax = sum of exponentials of logits
		* Requires a really good model as it relies on modeling the distribution of the outputs
* For our classification output, if we have both mutually exclusive labels and probabilities, we should use `tf.nn.softmax_cross_entropy_with_logits_v2()`. If the labels are mutually exclusive, but the probabilities aren’t, we should use `tf.nn.sparse_softmax_cross_entropy_with_logits()`. If our labels aren’t mutually exclusive, we should use `tf.nn.sigmoid_cross_entropy_with_logits()`.

#### Embeddings

* Think of families of models - embeddings in one model can be used to jumpstart embeddings in another model in the same family
* Embeddings - think of these like good, reusable software libraries
* Remember feature cross of hour of day and day of week for traffic predicitons.  Instead of one-hot encoding this, we could pass it to a dense layer, and then train the model to predict traffic as before.
	* Embeddings are weighted sum of feature cross values, weights learned from the data
	* Embedding learns how to embed the feature cross in lower-dimensional space (e.g., from 168-dimensional to only 2 real-valued numbers) using `tf.feature_column.embedding_column()`
		* `day_hour_embedding = tf.feature_column.embedding_column(day_hour_categorical_column, 2)`
	* Works with any categorical column, not just a feature cross
	* Can load checkpoint tensor from a checkpoint directory to jumpstart traffic model for another city, for example
* Transfer learning of embeddings from similar ML models (e.g., to predict TV viewership from traffic patterns - same latent factor)
	* First layer - the feature cross
	* Second layer - Mystery box labeled latent factor
	* Third layer - the embedding
	* Fourth layer
		* One side: image of traffic
		* Second side: Image of people watching TV
* Transfer learning might work on seemingly very different problems as long as they share the same latent factors
* Embeddings are useful for any categorical column
* Think of matrix of users and movies for Netflix - sparse ratings
	* We could organize by dimension of age appropriateness...
	* If we add a second dimension, e.g., gross ticket sales, it gives us more freedom in organizing movies by similarity
	* Adding more dimensions adds more distinctions, but overfitting can happen of course with too many
* If we originally had 500,000 movies, we go from a 500,000 dimensional space to a two-dimensional space, or D-dimensional, where D is the number of features used to represent a movie
* With 500,000 movies, you have to create 500,000 input nodes as one-hot encoded, - BAD IDEA
	* Dense representation is inefficient for storage and compute (takes up a ton of memory to do matrix math on such large matrices)
	* Dense tensor - anything where we store all the values for an input tensor - has nothing to do with the actual data in the tensor, just about how we're storing it
	* Store as sparse tensors, do matrix multiplication directly on sparse tensors
		* Build a dictionary mapping from each feature to an integer (e.g., each movie gets an integer) - for each user, just store the integers representing the movies the user has seen - this creates an efficient, sparse representation
		* `tf.feature_column.categorical_column_with_vocabulary_list()`, `tf.feature_column.categorical_column_with_identity()`, and `tf.feature_column.categorical_column_with_hash_bucket()` all store info as sparse vectors
* How does this really work?
* Embeddings are feature columns that function like layers - really no different than any other hidden layer
* Think of embeddings as a handy adapter that allows the network to incorporate sparse or categorical data well
* Can do this with regression, categorization, or ranking problem
* If a word is being represented by a three-dimensional embedding, each of the three nodes (in the embedding) can be considered as a dimension into which words are being projected. 
	* Edge weights between a movie and a hidden layer are the coordinate values in this lower dimensional projection.
* Think of MNIST - 28x28 images, 784 pixels - we only save the pixels with nonzero values (e.g., with part of the written digit), represent as sparse matrix
* Pass this to 3-dimensional embedding - these three numbers represent values of all pixels "turned on" in the image
	* This is what we're looking at in TensorBoard's "projections" tab, e.g., for T-SNE visualization
	* true for categorical variables, natural language text, and raw bitmaps - all of these are sparse
	* You'll see the clustering like in MNIST T-SNE projector if you have enough data and your model gets good accuracy
* When we use a linear or DNN classifier, output is a single logit if you only have one output / target class
* With MNIST, 10 labels (0-9) so 1 logit for each of the possible digits, logit layer
* If you use audio data, take embeddings of clips and compute Euclidian distance, that's a measure of similarity between songs
* Can also use embedding vectors as inputs to a clustering algorithm
* Can jointly embed diverse features - e.g., two different languages, text and corresponding audio
* Number of embeddings is hyperparameter - higher number can more accurately represent the relationship b/t input values, but more embeddings makes a larger model and slower training
* Good rule is to start with the fourth root of the total number of possible values, so for 500,000 movies, square root is about 700, square root of that is about 26, so start around 25 maybe
* 15-35 is rule of thumb, but not firm

* How to mix Keras and the Estimator API?  - once you have a compiled Keras model, you can get an Estimator.
	* `from tensorflow import keras`
	* `model = Sequential()`
	* ...
	* `estimator = keras.estimator.model_to_estimator(keras_model=model)`

* Then remove `model.fit()` and `model.evaulate()` from before, then use the new `estimator` as you did before, with training input function, train spec, eval spec, exporter, etc. and pass them into `train_and_evaluate()` function - this is how you productionize a Keras model
* Note: Linkage b/t input function and Keras model is through a naming convention
	* If you have a Keras layer named 'XYZ', you should have a feature named 'XYZ_input' returned in the features dict that comes out of your `train_input_fn()`
* https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf
* https://arxiv.org/abs/1712.00409
* Invest in data collection, augmentation, and enrichment, then top off with hyperparameter tuning
* http://colah.github.io/posts/2015-08-Understanding-LSTMs/ for the theory
* https://www.tensorflow.org/tutorials/recurrent for explanations
* https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb for sample code
* `tf.stack()` - creates a matrix from a list of lists - e.g. for time series / RNN data sets
* For this demo, he's using MSE as loss fx, SGD as optimizer
