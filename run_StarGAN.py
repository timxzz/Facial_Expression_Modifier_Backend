import os
import tensorflow as tf
import scipy.misc as scm
from StarGAN import StarGAN

''' config settings '''

project_name = "StarGAN_Face_1_"
train_flag = False

'''-----------------'''

# gpu_number = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" #args.gpu_number

# with tf.device('/gpu:{0}'.format(gpu_number)):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
#     config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

input_image = scm.imread('./202593.jpg')
print(input_image.shape)

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        model = StarGAN(sess, project_name)
        out_images = model.test(input_image, train_flag)

        for i in range(model.data_label_vector_size):
            # output save
            scm.imsave('./result/' + str(i) + '_celebra_output.png', out_images[i])