# initializer 总结

## initialzier in tensorflow

### variance_scaling

[tf.keras.initializers.VarianceScaling](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling)

aliases:

- tf.initializers.variance_scaling
- tf.keras.initializers.VarianceScaling
- tf.variance_scaling_initializer



### glorot_uniform

If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

inherits from variance_scaling

same as `variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')`

tf default kernel_initializer is [glorot_uniform_initializer](https://www.tensorflow.org/api_docs/python/tf/glorot_uniform_initializer)

It draws samples from a uniform distribution within `[-limit, limit]` where limit is `sqrt(6 / (fan_in + fan_out))` where `fan_in` is the number of input units in the weight tensor and `fan_out` is the number of output units in the weight tensor.

### glorot_normal

inherits from variance_scaling

