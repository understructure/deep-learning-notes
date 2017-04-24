# Notes on Deep Learning (ud17)


### Some Links

http://scikit-learn.org/stable/

https://en.wikipedia.org/wiki/Perceptron

https://en.wikipedia.org/wiki/Gradient_descent

http://neuralnetworksanddeeplearning.com/chap2.html

http://www.numpy.org/

https://www.tensorflow.org/


`conda create -n py27 python=2.7
conda env export > environment.yaml

pip freeze > requirements.txt # for people not using conda`



https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/

http://conda.pydata.org/docs/using/index.html

https://docs.python.org/3.6/whatsnew/3.6.html#pep-498-formatted-string-literals

http://ipython.readthedocs.io/en/stable/interactive/magics.html


------------------------------------------------------------------------------------------------------------------------
To convert it and immediately see it, use:

	jupyter nbconvert notebook.ipynb --to slides --post serve

This will open up the slideshow in your browser so you can present it.
------------------------------------------------------------------------------------------------------------------------


* https://github.com/lengstrom/fast-style-transfer
* https://www.youtube.com/watch?v=LoePx3QC5Js
* http://selfdrivingcars.mit.edu/deeptrafficjs/
* https://www.manning.com/books/grokking-deep-learning (not totally available yet - code available for discount)
* http://neuralnetworksanddeeplearning.com/
* http://www.deeplearningbook.org/
* http://scikit-learn.org/stable/tutorial/basic/tutorial.html
* http://matplotlib.org/users/pyplot_tutorial.html
* http://pandas.pydata.org/pandas-docs/stable/10min.html#min

Gradient Descent - Technique used to minimize loss function over time
Numpy - *the* matrix multiplication library for Python
Learning rate - hyperparameter - tuning knob - how fast model learns


Gradient is essentially a tangent - bowl-like curve
Gradient just tells us what to do to get our values closer to minimizing error - add, subtract, multiply, etc.
If a function is differentiable, we know we can optimize it
To do this, calculate partial derivative w.r.t. our values (b and m)
	- calculate derivative for each, without knowledge of the other

A simple trask for this course

Hyperparameters - high-level tuning parameters of the network, helps determine:

- how fast model runs
- # of hidden layers
- # of neurons

Random Search

- pick a range, let it randomly sample from uniform distribution of those values

Initialize weights - from normal distribution w/ low deviation (values pretty close together)
values from -1.0 to 1.0


apply weight values to input data - dot product (two matrixes multiplyed)


functions to check out (probably all in numpy):

`nonlin  
np.dot # dot product of two matrixes  
np.exp # exponent?`  


check error - how far off is this from actual value?
don't change input data, DO change weights

Gradient = Slope
Calculate the gradient of error w.r.t. weight values - use the derivative
Essentially calculating Gradient of sigmoid function at a given function

Picture dropping a ball into a convex function - if gradient is negative, move to right, if positive, move to left - use gradient to update weights accordingly each time

Repeat until gradient is 0, which gives us smallest error value

We're descending our gradient to approach 0, and using it to update our weight values iteratively 

To do this programmatically, multiply derivative by the error - this gives us our Error Weighted Derivative, or L2 Delta

L2 Delta - matrix of values, one for each predicted output - gives us a direction

Multiply the L2 Delta values by transpose of its associated weight matrix to get previous layer's error, then use that error to do same operation as before to get direction values to update the associated layer's weights so error is minimized

Lastly, we'll update the weight matrixes for each associated layer by multiplying them by their respective delta

1. Deep learning uses:  
	- Linear Algebra  
	- Statistics, and  
	- Calculus  
2. A Neural Net performs a series of operations on an input tensor to make a prediction
3. We can optimize a prediction using gradient descent to backpropogate errors and updating weights accordingly



