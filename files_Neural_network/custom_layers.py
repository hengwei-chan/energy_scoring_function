from keras.layers import Layer
from keras.initializers import Constant
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops
from keras.constraints import NonNeg

def _to_tensor(x, dtype):
    return ops.convert_to_tensor(x, dtype=dtype)

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(1, input_shape[1]), initializer=Constant(value=3), name = 'kernel', constraint = NonNeg())
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        output = -x + self.kernel
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class MyLayer_Activation(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer_Activation, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(1, 1), initializer=Constant(value=20), name = 'kernel', constraint = NonNeg())
        super(MyLayer_Activation, self).build(input_shape)

    def call(self, x):
        x = self.kernel*x + 0.5
        zero = _to_tensor(0., x.dtype.base_dtype)
        one = _to_tensor(1., x.dtype.base_dtype)
        output = clip_ops.clip_by_value(x, zero, one)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def custom_activation1(x):
    x = 10*x + 0.5
    zero = _to_tensor(0., x.dtype.base_dtype)
    one = _to_tensor(1., x.dtype.base_dtype)
    x = clip_ops.clip_by_value(x, zero, one)
    return x

def custom_activation2(x):
    x = 50*x + 0.5
    zero = _to_tensor(0., x.dtype.base_dtype)
    one = _to_tensor(1., x.dtype.base_dtype)
    x = clip_ops.clip_by_value(x, zero, one)
    return x


