import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2, inception
slim = tf.contrib.slim

class Resnet_V2_101_Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      _, end_points = resnet_v2.resnet_v2_101(x_input, 
          num_classes=self.num_classes, is_training=False, reuse=reuse)
    self.built = True
    output = end_points['predictions']

    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs

class Inception_V3_Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(x_input, 
          num_classes=self.num_classes, is_training=False, reuse=reuse)
    self.built = True
    output = end_points['Predictions']

    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs