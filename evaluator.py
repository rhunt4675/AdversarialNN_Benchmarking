#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import imageio
import models
import json
import time
import os

from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MomentumIterativeMethod, \
        SaliencyMapMethod, MadryEtAl, LBFGS
from cleverhans.model import CallableModelWrapper
slim = tf.contrib.slim

tf.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
tf.flags.DEFINE_string('checkpoint_path', '', 'Path to checkpoint for ResNet network.')
tf.flags.DEFINE_string('input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string('output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_string('model_arch', 'resnet_v2_101', 'Model architecture to use: (resnet_v2_101/inception_v3)')
tf.flags.DEFINE_string('attack_type', 'FGSM', 'Cleverhans attack to run: (FGSM/BIM/MIM/PGD/JSMA/LBFGS).')
tf.flags.DEFINE_integer('image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 224, 'Height of each input images.')
tf.flags.DEFINE_integer('batch_size', 16, 'How many images process at one time.')
tf.flags.DEFINE_boolean('show_predictions', False, '[Debug] Print model predictions of clean and adversarial images.')
FLAGS = tf.flags.FLAGS

def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.
  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """

  height = batch_shape[1]
  width = batch_shape[2]
  images = np.zeros(batch_shape)

  filenames = []
  idx = 0
  batch_size = batch_shape[0]

  with tf.Session() as sess:
    sess.run((tf.local_variables_initializer(), tf.global_variables_initializer()))

    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.JPEG')):
      image = imageio.imread(filepath)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image = tf.expand_dims(image, [0])
      image = tf.image.grayscale_to_rgb(tf.expand_dims(image, -1)) if sess.run(tf.rank(image)) == 3 else image
      image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
      image = tf.squeeze(image, [0])
#      image = tf.subtract(image, 0.5)
#      image = tf.multiply(image, 2.0)
      image = sess.run(image)

      # Images for inception classifier are normalized to be in [-1, 1] interval.
      images[idx, :, :, :] = image
      filenames.append(os.path.basename(filepath))
      idx += 1
      if idx == batch_size:
        yield filenames, images
        filenames = []
        images = np.zeros(batch_shape)
        idx = 0
    if idx > 0:
      yield filenames, images

def save_images(images, filenames, output_dir):
  """Saves images to the output directory.
  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i,_ in enumerate(filenames):
    # img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
    img = (images[i, :, :, :] * 255.0).astype(np.uint8)
    imageio.imwrite(os.path.join(output_dir, filenames[i]), img)

def main(_):
  tf.logging.set_verbosity(tf.logging.DEBUG)

  # Images for inception classifier are normalized to be in [-1, 1] interval,
  num_classes = 1001
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  # Load ImageNet Class Labels
  with open('labels.json') as f:
    labels = json.load(f)

  # Prepare Graph
  with tf.Graph().as_default():

    # Build Model
    if FLAGS.model_arch.lower() == 'resnet_v2_101':
      model = models.Resnet_V2_101_Model(num_classes)
      exceptions = []

    elif FLAGS.model_arch.lower() == 'inception_v3':
      model = models.Inception_V3_Model(num_classes)
      exceptions = ['InceptionV3/AuxLogits.*']

    else:
      raise ValueError('Invalid model architecture specified: {}'.format(FLAGS.model_arch))

    # Define Model Variables
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    FastGradientMethod(model).generate(x_input)
    model_variables = tf.contrib.framework.filter_variables(
      slim.get_model_variables(), exclude_patterns=exceptions)

    # Load Session
    saver = tf.train.Saver(model_variables)
    with tf.train.SessionManager().prepare_session(master=FLAGS.master, 
          checkpoint_filename_with_path=FLAGS.checkpoint_path, saver=saver) as sess:

      # For Targeted Attacks
      target_idx = 0 # This will vary
      target = tf.constant(0, shape=[FLAGS.batch_size, num_classes])
#      target = np.zeros((FLAGS.batch_size, num_classes), dtype=np.uint32)
#      target[:, target] = 1
      
      # Build Attack
      if FLAGS.attack_type.lower() == 'fgsm':
        fgsm_opts = {'eps': 0.3, 'clip_min': 0, 'clip_max': 1., 'y_target': None}
        fgsm = FastGradientMethod(model)
        x_adv = fgsm.generate(x_input, **fgsm_opts)
      
      elif FLAGS.attack_type.lower() == 'bim':
        bim_opts = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1., 'y_target': None}
        bim = BasicIterativeMethod(model)
        x_adv = bim.generate(x_input, **bim_opts)
      
      elif FLAGS.attack_type.lower() == 'mim':
        mim_opts = {'eps': 0.3, 'clip_min': 0, 'clip_max': 1.}
        mim = MomentumIterativeMethod(model)
        x_adv = mim.generate(x_input, **mim_opts)
      
      elif FLAGS.attack_type.lower() == 'pgd':
        pgd_opts = {'eps': 0.3, 'clip_min': 0, 'clip_max': 1.}
        pgd = MadryEtAl(model)
        x_adv = pgd.generate(x_input, **pgd_opts)

      # Broken
      elif FLAGS.attack_type.lower() == 'jsma':
        jsma_opts = {'theta': 1., 'gamma': 0.1, 'clip-min': 0., 'clip-max': 1., 'y_target': None}
        jsma = SaliencyMapMethod(model)
        x_adv = jsma.generate(x_input, **jsma_opts)

      elif FLAGS.attack_type.lower() == 'lbfgs':
        lbfgs_opts = {'y_target': target}
        lbfgs = LBFGS(model)
        x_adv = lbfgs.generate(x_input, **lbfgs_opts)

      else:
        raise ValueError('Invalid attack type specified: {}'.format(FLAGS.attack_type))

      start_time, batch_time, num_processed = time.time(), time.time(), 0
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(adv_images, filenames, FLAGS.output_dir)

        if FLAGS.show_predictions:
          preds = sess.run(model(np.float32(images)))
          probs = np.amax(preds, axis=1)
          classes = np.argmax(preds, axis=1)
          adv_preds = sess.run(model(adv_images))
          adv_probs = np.amax(adv_preds, axis=1)
          adv_classes = np.argmax(adv_preds, axis=1)

          for i,_ in enumerate(filenames):
            print('\nOriginal: {:.2f}% ({})\nAdversarial: {:.2f}% ({})'.format( \
              probs[i]*100, labels[str(classes[i])], adv_probs[i]*100, labels[str(adv_classes[i])]))

        time_delta = time.time() - batch_time
        batch_time = time.time()
        num_processed += len(filenames)
        print('[SPEED ESTIMATION] BatchRate={:.4f} Hz; AverageRate={:.4f} Hz'.format( \
          (len(filenames) / time_delta * 1.0), ((num_processed * 1.0) / (batch_time - start_time))))

if __name__ == '__main__':
  tf.app.run()
