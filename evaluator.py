#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import imageio
import models
import json
import os

from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MomentumIterativeMethod, \
        SaliencyMapMethod, VirtualAdversarialMethod, CarliniWagnerL2, ElasticNetMethod, DeepFool, \
        MadryEtAl, SPSA
from cleverhans.model import CallableModelWrapper
slim = tf.contrib.slim

tf.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
tf.flags.DEFINE_string('checkpoint_path', '', 'Path to checkpoint for ResNet network.')
tf.flags.DEFINE_string('input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string('output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_string('model_arch', 'resnet_v2_101', 'Model architecture to use: (resnet_v2_101/inception_v3)')
tf.flags.DEFINE_string('attack_type', 'FGSM', 'Cleverhans attack to run: (FGSM/BIM/MIM/JSMA/VAM/CWL2/ENM/DF/MADRY/SPSA).')
tf.flags.DEFINE_integer('image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 224, 'Height of each input images.')
tf.flags.DEFINE_integer('batch_size', 16, 'How many images process at one time.')
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

  """images = np.zeros(batch_shape)
  filenames = []
  idx = 0

  batch_size = batch_shape[0]
  height = batch_shape[1]
  width = batch_shape[2]

  jpegs = tf.train.match_filenames_once(os.path.join(input_dir, '*.jpg'))
  num_images = tf.size(jpegs)
  filename_queue = tf.train.string_input_producer(jpegs)
  image_reader = tf.WholeFileReader()

  with tf.Session() as sess:
    sess.run((tf.local_variables_initializer(), tf.global_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(sess.run(num_images)):
      file_name, image_file = image_reader.read(filename_queue)
      image = tf.image.decode_jpeg(image_file)

      image_tensor = inception_preprocessing.preprocess_image(image, height, width)
      images[idx, :, :, :] = sess.run(image_tensor)
      filenames.append(os.path.basename(sess.run(file_name).decode('UTF-8')))

      idx += 1
      if idx == batch_size:
        yield filenames, images
        filenames = []
        images = np.zeros(batch_shape)
        idx = 0

    coord.request_stop()
    coord.join(threads)
  if idx > 0:
    yield filenames, images"""

  height = batch_shape[1]
  width = batch_shape[2]
  images = np.zeros(batch_shape)

  filenames = []
  idx = 0
  batch_size = batch_shape[0]

  with tf.Session() as sess:
    sess.run((tf.local_variables_initializer(), tf.global_variables_initializer()))

    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg')):
      image = imageio.imread(filepath)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image = tf.expand_dims(image, [0])
      image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
      image = tf.squeeze(image, [0])
      image = tf.subtract(image, 0.5)
      image = tf.multiply(image, 2.0)
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
    img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
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

      # Build Attack
      if FLAGS.attack_type.lower() == 'fgsm':
        fgsm_opts = {'eps': 1/8.0, 'clip_min': -1., 'clip_max': 1.}
        fgsm = FastGradientMethod(model)
        x_adv = fgsm.generate(x_input, **fgsm_opts)
      
      elif FLAGS.attack_type.lower() == 'bim':
        bim_opts = {'eps': 1/8.0, 'eps_iter': 1/16.0, 'nb_iter': 5, 'clip_min': -1., 'clip_max': 1.}
        bim = BasicIterativeMethod(model)
        x_adv = bim.generate(x_input, **bim_opts)
      
      elif FLAGS.attack_type.lower() == 'mim':
        mim_opts = {'eps': 1/8.0, 'eps_iter': 1/16.0, 'nb_iter': 5, 'clip_min': -1., 'clip_max': 1.}
        mim = MomentumIterativeMethod(model)
        x_adv = mim.generate(x_input, **mim_opts)
      
      # Broken
      elif FLAGS.attack_type.lower() == 'jsma':
        jsma_opts = {'clip_min': -1., 'clip_max': 1.}
        jsma = SaliencyMapMethod(model)
        x_adv = jsma.generate(x_input, **jsma_opts)

      elif FLAGS.attack_type.lower() == 'vam':
        vam_opts = {'eps': 1/8.0, 'clip_min': -1., 'clip_max': 1.}
        vam = VirtualAdversarialMethod(model)
        x_adv = vam.generate(x_input, **vam_opts)

      elif FLAGS.attack_type.lower() == 'cwl2':
        cwl2_opts = {'clip_min': -1., 'clip_max': 1.}
        cwl2 = CarliniWagnerL2(model)
        x_adv = cwl2.generate(x_input, **cwl2_opts)

      elif FLAGS.attack_type.lower() == 'enm':
        enm_opts = {'clip_min': -1., 'clip_max': 1.}
        enm = ElasticNetMethod(model)
        x_adv = enm.generate(x_input, **enm_opts)

      elif FLAGS.attack_type.lower() == 'df':
        df_opts = {'clip_min': -1., 'clip_max': 1.}
        df = DeepFool(model)
        x_adv = df.generate(x_input, **df_opts)

      elif FLAGS.attack_type.lower() == 'madry':
        madry_opts = {'eps': 1/8.0, 'eps_iter': 1/16.0, 'nb_iter': 5, 'clip_min': -1., 'clip_max': 1.}
        madry = MadryEtAl(model)
        x_adv = madry.generate(x_input, **madry_opts)

      elif FLAGS.attack_type.lower() == 'spsa':
        spsa_opts = {'epsilon': 1/8.0}
        spsa = SPSA(CallableModelWrapper(model, 'probs'))
        x_adv = spsa.generate(x_input, **spsa_opts)

      else:
        raise ValueError('Invalid attack type specified: {}'.format(FLAGS.attack_type))


      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(adv_images, filenames, FLAGS.output_dir)

        preds = sess.run(model(np.float32(images)))
        probs = np.amax(preds, axis=1)
        classes = np.argmax(preds, axis=1)
        adv_preds = sess.run(model(adv_images))
        adv_probs = np.amax(adv_preds, axis=1)
        adv_classes = np.argmax(adv_preds, axis=1)

        for i,_ in enumerate(filenames):
          print('\nOriginal: {:.2f}% ({})\nAdversarial: {:.2f}% ({})'.format( \
            probs[i]*100, labels[str(classes[i])], adv_probs[i]*100, labels[str(adv_classes[i])]))

if __name__ == '__main__':
  tf.app.run()