

### Troubleshooting Tips


If you do something like this:

<pre>
a = tf.placeholder(tf.int16, name="AAA")
b = tf.placeholder(tf.int16, name="BBB")
opz = a * b
sess.run(opz, {a: 10, b:20})
</pre>
This works fine, but if you try to redefine a or b and then run a session with anything that refers to either, it will throw errors:

<pre>
a = tf.placeholder(tf.int16, name="AAA2")

# This will throw an error:
sess.run(opz, {a: 10, b:20})```
</pre>
The fix is to redefine the operation `opz` to point to the new `a` object:
<pre>
opz = a * b
sess.run(opz, {a: 10, b:20})
</pre>

That will make the errors go away.


