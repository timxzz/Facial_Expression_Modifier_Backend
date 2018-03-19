import glob
import time
import datetime
import random
import tensorflow as tf
import numpy as np

from op_base import op_base
from nn_module import l1_loss
from util import get_image, get_label, inverse_image

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits

class Operator(op_base):
    def __init__(self, sess, project_name):
        op_base.__init__(self, sess, project_name)
        self.build_model()

    def build_model(self):
        # Input placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.data_size, self.data_size, 3], name='x')
        self.x_c = tf.placeholder(tf.float32, shape=[None, self.data_label_vector_size], name='x_c')
        self.target = tf.placeholder(tf.float32, shape=[None, self.data_size, self.data_size, 3], name='target')
        self.target_c = tf.placeholder(tf.float32, shape=[None, self.data_label_vector_size], name='target_c')
        self.alpha = tf.placeholder(tf.float32, shape=[None, 1], name='alpha')
        self.lr = tf.placeholder(tf.float32, name='lr')

        # Generator
        self.G_f = self.generator(self.x, self.target_c)
        self.G_recon = self.generator(self.G_f, self.x_c, reuse=True)
        
        self.G_test = self.generator(self.x, self.target_c, reuse=True)
        
        # Discriminator
        self.D_f, self.D_f_cls = self.discriminator(self.G_f)
        self.D_target, self.D_target_cls = self.discriminator(self.target, reuse=True) # discriminate with the target
        
        # Gradient Penalty
        self.real_data = tf.reshape(self.target, [-1, self.data_size*self.data_size*3]) # interpolate with target
        self.fake_data = tf.reshape(self.G_f, [-1, self.data_size*self.data_size*3])
        self.diff = self.fake_data - self.real_data
        self.interpolate = self.real_data + self.alpha*self.diff
        
        self.inter_reshape = tf.reshape(self.interpolate, [-1, self.data_size, self.data_size, 3])
        self.G_inter, _ = self.discriminator(self.inter_reshape, reuse=True)
        
        self.grad = tf.gradients(self.G_inter, 
                                 xs=[self.inter_reshape])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.grad), axis=[1,2,3]))
        self.gp = tf.reduce_mean(tf.square(self.slopes - 1.))
        
        
        # Wasserstein loss
        self.wd = tf.reduce_mean(self.D_target) - tf.reduce_mean(self.D_f)
        self.L_adv_D = -self.wd + self.gp * self.lambda_gp
        self.L_adv_G = -tf.reduce_mean(self.D_f)
        
        self.L_D_cls = tf.reduce_mean(cross_entropy(labels=self.target_c, logits=self.D_target_cls))# discriminate with the target
        self.L_G_cls = tf.reduce_mean(cross_entropy(labels=self.target_c, logits=self.D_f_cls))
        self.L_G_recon = l1_loss(self.x, self.G_recon)
                
        self.L_D = self.L_adv_D + self.lambda_cls * self.L_D_cls
        self.L_G = self.L_adv_G + self.lambda_cls * self.L_G_cls + self.lambda_recon * self.L_G_recon
        

        # Variables
        D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminator")
        G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator")

        # Optimizer
        self.opt_D = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.L_D, var_list=D_vars)
        self.opt_G = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.L_G, var_list=G_vars)

        # initializer
        self.sess.run(tf.global_variables_initializer())

        # tf saver
        self.saver = tf.train.Saver(max_to_keep=(self.max_to_keep))

        try:
            self.load(self.sess, self.saver, self.ckpt_dir)
        except:
            print("Error: Could not load model.")

    

    def test(self, data, train_flag=True):
        print('Test Sample Generation...')
        # generate output
        in_img_num = 1
        img_size = self.data_size
        gen_img_num = self.data_label_vector_size
        label_size = self.data_label_vector_size
        
        test_data = [get_image(data)]
        test_data = np.repeat(test_data, [label_size]*in_img_num, axis=0)
        
        # get one-hot labels
        int_labels = list(range(label_size))
        one_hot = np.zeros((label_size, label_size))
        one_hot[np.arange(label_size), int_labels] = 1.0
        target_labels = one_hot
        
        
        output_gen = (self.sess.run(self.G_test, feed_dict={self.x: test_data, 
                                                            self.target_c: target_labels}))  # generator output

        output_gen = [inverse_image(output_gen[i]) for i in range(gen_img_num)]

        return output_gen

       
        
    def test_expr(self, train_flag=True):
        print('Train Sample Generation...')
        # generate output
        img_num =  36 #self.batch_size
        display_img_num = int(img_num / 3)
        img_size = self.data_size

        output_f = int(np.sqrt(img_num))
        im_output_gen = np.zeros([img_size * output_f, img_size * output_f, 3])
        
        # load data
        data_path = self.data_dir

        if os.path.exists(data_path + '.npy'):
            data = np.load(data_path + '.npy')
        else:
            data = sorted(glob.glob(os.path.join(data_path, "*.*")))
            data = pair_expressions(data)
            np.save(data_path + '.npy', data)

        # Test data shuffle
        random_order = np.random.permutation(len(data))
        data = [data[i] for i in random_order[:]]
        
        batch_files = data[0: display_img_num]
        test_inputs = [get_image(batch_file[0]) for batch_file in batch_files]
        test_inputs_o = [scm.imread((batch_file[0])) for batch_file in batch_files]
        test_targets = [scm.imread((batch_file[1])) for batch_file in batch_files]
        test_target_labels = [get_label(batch_file[1], self.data_label_vector_size) for batch_file in batch_files]

        output_gen = (self.sess.run(self.G_test, feed_dict={self.x: test_inputs, 
                                                            self.target_c: test_target_labels}))  # generator output

        output_gen = [inverse_image(output_gen[i]) for i in range(display_img_num)]

        for i in range(output_f): # row
            for j in range(output_f): # col
                if j % 3 == 0: # input img
                    im_output_gen[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :] \
                        = test_inputs_o[int(j / 3) + (i * int(output_f / 3))]
                elif j % 3 == 1: # output img
                    im_output_gen[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :] \
                        = output_gen[int(j / 3) + (i * int(output_f / 3))]
                else: # target img
                    im_output_gen[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :] \
                        = test_targets[int(j / 3) + (i * int(output_f / 3))]
                   

        labels = np.argmax(test_target_labels, axis=1)
        label_string = ''.join(str(int(l)) for l in labels)
        # output save
        scm.imsave(self.project_dir + '/result/' + str(self.count) + '_' + label_string 
                   + '_expr_output.bmp', im_output_gen)
