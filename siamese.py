import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
#import tensorflow.contrib.slim.nets as nets
#import d
#import vgg

#s = tf.Session(config=tf.ConfigProto(gpu_options={'allow_growth':True}))

#images = tf.placeholder(tf.float32, [None, 224, 224, 3])
#predictions = d.vgg_16(VGG_inputs=images)
#variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])



def vgg_arg_scope(weight_decay=0.0005,reuse=False):

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer(),
                      ):
    with slim.arg_scope([slim.conv2d], padding='SAME',reuse=tf.AUTO_REUSE) as arg_sc:
      return arg_sc


def vgg_16(inputs,is_training=True,dropout_keep_prob=0.5,scope='vgg_16',fc_conv_padding='VALID'):

 with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.flatten(net)
      print net.op.name
      print net.shape
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      return net, end_points


if __name__=='__main__':
	with tf.name_scope('similarity'):
	    label = tf.placeholder(tf.int32,[None,1])
	    label = tf.to_float(label)

	margin=0.2

	anchor_inputs = tf.placeholder(tf.float32,[None,224,224,3],name='Anchor_Placeholder')
	positive_inputs = tf.placeholder(tf.float32,[None,224,224,3],name='Positive_Placeholder')
	with slim.arg_scope(vgg_arg_scope()):
	  positive, net1 = vgg_16(positive_inputs)
	with slim.arg_scope(vgg_arg_scope(reuse=True)):
	  anchor  , net2 = vgg_16(anchor_inputs)

	#print net2
	#for key,value in net2.items():
	#   print '{}\t : {}'.format(key,value.shape)


	variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
	#variables_to_restore = slim.get_variables_to_restore()
	init_assign_op, init_feed_dict = slim.assign_from_checkpoint('./vgg_16.ckpt', variables_to_restore)


	print 'LOSS.........................................................'
	with tf.name_scope('contrastive-loss'):
	   loss = tf.contrib.losses.metric_learning.contrastive_loss(labels=label,embeddings_anchor=anchor,embeddings_positive=positive,margin=margin)
	   print 'contrastive-loss::',loss.op.name

	with tf.name_scope('Prediction'):
	   pred = tf.norm(positive-anchor,ord='euclidean')
	   #pred = tf.sqrt(tf.reduce_mean(tf.square(positive-anchor),1))

	with tf.Session(config=tf.ConfigProto(gpu_options={'allow_growth':True})) as s:
	  s.run(tf.global_variables_initializer())
	  s.run(init_assign_op, init_feed_dict)

	  # Save as graph
	  #out_node = [[(n.name) for n in tf.get_default_graph().as_graph_def().node][-2]]
	  #out_node = ['vgg_16_1/Flatten/flatten/Reshape','vgg_16/Flatten/flatten/Reshape']
	  out_node = ['contrastive-loss/contrastive_loss']
	  output_graph_def = graph_util.convert_variables_to_constants(
	  s,
	  tf.get_default_graph().as_graph_def(),
	  out_node)
	#print [(n.name) for n in tf.get_default_graph().as_graph_def().node]

	model_file = './siam.pb'
	with tf.gfile.GFile(model_file,'wb') as f:
	  f.write(output_graph_def.SerializeToString())
	#print [(n.name) for n in tf.get_default_graph().get_operations()]

        

