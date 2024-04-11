import os
import numpy as np
import subprocess
import tensorflow as tf
from utils import utils
from utils.network import DialUNet as UNet

class InferService:
    def __init__(self, ckpt_path, gpu_flag, debug_flag):
        # if gpu_flag is enabled, we assume that the target is running with a linux machine
        # note that if we are running the service on a mac silicon arch, this flag should be disabled
        if gpu_flag:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(
                [int(x.split()[2]) for x in subprocess.Popen(
                    "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE,
                ).stdout.readlines()]
            ))
        else:    
            os.environ["CUDA_VISIBLE_DEVICES"] = ''

        # set up the model and define the graph
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            tf.compat.v1.disable_eager_execution()
            self.input = tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,5])
            reflection = tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,5])
            target = tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,5])
            self.overexp_mask = utils.tf_overexp_mask(self.input)
            self.tf_input, self.tf_reflection, _, real_input = utils.prepare_real_input(
                self.input, target, reflection, self.overexp_mask)
            self.reflection_layer = UNet(real_input, ext='Ref_')
            self.transmission_layer = UNet(tf.concat([real_input, self.reflection_layer],axis=3),ext='Tran_') 

        # Load session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        var_restore = [v for v in tf.compat.v1.trainable_variables()]
        saver_restore = tf.compat.v1.train.Saver(var_restore)
        if debug_flag:
            for var in tf.compat.v1.trainable_variables():
                print("Listing trainable variables ... ")
                print(var)

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        assert ckpt
        saver_restore = tf.compat.v1.train.Saver([var for var in tf.compat.v1.trainable_variables()])
        if debug_flag:
            print('loaded ' + ckpt.model_checkpoint_path)
        saver_restore.restore(self.sess, ckpt.model_checkpoint_path)

    # Predict function
    def predict(self, image):
        h,w = utils.crop_shape(image)

        _, pred_image_t, _, _ = self.sess.run(
            [self.overexp_mask, self.transmission_layer, self.reflection_layer, self.tf_input],
            feed_dict={self.input:image[:,:h,:w,:]}
        )

        img = np.uint16((0.5*pred_image_t[0,:,:,4]).clip(0,1)*65535.0)
        return img
