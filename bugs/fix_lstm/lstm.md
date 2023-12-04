# Output after init_between_ops

## Command to run
python -m pytest tests/explainers/test_deep.py::test_tf_keras_mnist_cnn

## Breakpoint after self._init_between_tensors(self.model_output.op, self.model_inputs)
(Pdb++) self.between_ops
[<tf.Operation 'conv2d/Conv2D' type=Conv2D>, <tf.Operation 'conv2d/BiasAdd' type=BiasAdd>, <tf.Operation 'conv2d/Relu' type=Relu>, <tf.Operation 'conv2d_1/Conv2D' type=Conv2D>, <tf.Operation 'conv2d_1/BiasAdd' type=BiasAdd>, <tf.Operation 'conv2d_1/Relu' type=Relu>, <tf.Operation 'max_pooling2d/MaxPool' type=MaxPool>, <tf.Operation 'dropout/cond' type=If>, <tf.Operation 'dropout/cond/Identity' type=Identity>, <tf.Operation 'flatten/Reshape' type=Reshape>, <tf.Operation 'dense/MatMul' type=MatMul>, <tf.Operation 'dense/BiasAdd' type=BiasAdd>, <tf.Operation 'dense/Relu' type=Relu>, <tf.Operation 'dropout_1/cond' type=If>, <tf.Operation 'dropout_1/cond/Identity' type=Identity>, <tf.Operation 'dense_1/MatMul' type=MatMul>, <tf.Operation 'dense_1/BiasAdd' type=BiasAdd>]
(Pdb++) self.between_tensors
{
    'conv2d/Conv2D:0': True,
    'conv2d/BiasAdd:0': True,
    'conv2d/Relu:0': True,
    'conv2d_1/Conv2D:0': True,
    'conv2d_1/BiasAdd:0': True,
    'conv2d_1/Relu:0': True,
    'max_pooling2d/MaxPool:0': True,
    'dropout/cond:0': True,
    'dropout/cond:1': True,
    'dropout/cond:2': True,
    'dropout/cond:3': True,
    'dropout/cond:4': True,
    'dropout/cond:5': True,
    'dropout/cond:6': True,
    'dropout/cond:7': True,
    'dropout/cond/Identity:0': True,
    'flatten/Reshape:0': True,
    'dense/MatMul:0': True,
    'dense/BiasAdd:0': True,
    'dense/Relu:0': True,
    'dropout_1/cond:0': True,
    'dropout_1/cond:1': True,
    'dropout_1/cond:2': True,
    'dropout_1/cond:3': True,
    'dropout_1/cond:4': True,
    'dropout_1/cond:5': True,
    'dropout_1/cond:6': True,
    'dropout_1/cond:7': True,
    'dropout_1/cond/Identity:0': True,
    'dense_1/MatMul:0': True,
    'dense_1/BiasAdd:0': True,
    'conv2d_input:0': True
    }
(Pdb++) 