import glob
import time
import numpy as np
import tensorflow as tf
import os

class op_base:
    def __init__(self, sess, project_name):
        self.sess = sess

        # Train
        self.flag = True #args.flag
        self.gpu_number = 0 #args.gpu_number
        self.project = project_name  #args.project

        # Train Data
        self.data_dir = "./Face_data/Faces_with_expression_label/dataset_128x128" #args.data_dir
        self.dataset = "expr" #args.dataset  # celeba
        self.data_size = 128 #args.data_size  # 64 or 128
        self.data_opt = "crop" #args.data_opt  # raw or crop
        self.data_label_vector_size = 7 #size of one-hot-encoded label vector

        # Train Iteration
        self.niter = 200 #50 #args.niter # Epoch
        self.niter_snapshot = 500 #args.nsnapshot
        self.max_to_keep = 50 #args.max_to_keep models

        # Train Parameter
        self.batch_size = 16 #args.batch_size
        self.learning_rate = 1e-4 #args.learning_rate
        self.mm = 0.5 #args.momentum
        self.mm2 = 0.999 #args.momentum2
        self.input_size = 128 #args.input_size
        
        self.lambda_cls = 1.
        self.lambda_recon = 10.
        self.lambda_gp = 10.

        

        # Result Dir & File
        self.project_dir = './'
        self.ckpt_dir = os.path.join(self.project_dir, 'models')
        self.model_name = "{0}.model".format(self.project)
        self.ckpt_model_name = os.path.join(self.ckpt_dir, self.model_name)

    def load(self, sess, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_dir, ckpt_name))