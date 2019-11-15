import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_circles

attack_config = {'max_distance': 0.5, 'num_steps': 10, 
                 'step_size': 0.05, 'random_start': True,
                 'norm': 'Linf', 'optimizer': 'adam',
                 'x_min': 0.0, 'x_max': 1.0}

class PGDAttack:
    """Base class for various attack methods"""
    def __init__(self, max_distance, num_steps, step_size, random_start, x_min,
                 x_max, batch_size, norm, optimizer, input_size=2):
        self.max_distance = max_distance
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.x_min = x_min
        self.x_max = x_max
        self.norm = norm
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.delta = tf.Variable(np.zeros((batch_size, input_size)),
                                 dtype=tf.float32,
                                 name='delta')
        self.x0 = tf.Variable(np.zeros((batch_size, input_size)),
                              dtype=tf.float32,
                              name='x0')
        self.y = tf.Variable(np.zeros(batch_size), dtype=tf.int64, name='y')
        self.c_constants = tf.Variable(np.zeros(batch_size),
                                       dtype=tf.float32,
                                       name='c_constants')

        self.delta_input = tf.placeholder(dtype=tf.float32,
                                          shape=[batch_size, input_size],
                                          name='delta_input')
        self.x0_input = tf.placeholder(dtype=tf.float32,
                                       shape=[batch_size, input_size],
                                       name='x0_input')
        self.y_input = tf.placeholder(dtype=tf.int64,
                                      shape=[batch_size],
                                      name='delta_input')
        self.c_constants_input = tf.placeholder(dtype=tf.float32,
                                                shape=[batch_size],
                                                name='c_constants_input')

        self.assign_delta = self.delta.assign(self.delta_input)
        self.assign_x0 = self.x0.assign(self.x0_input)
        self.assign_y = self.y.assign(self.y_input)
        self.assign_c_constants = self.c_constants.assign(
            self.c_constants_input)

        self.x = self.x0 + self.delta
        ord = {'L2': 2, 'Linf': np.inf}[norm]
        self.dist = tf.norm(self.x - self.x0, ord=ord, axis=1)

    def setup_optimizer(self):
        if self.optimizer == 'adam':
            # Setup the adam optimizer and keep track of created variables
            start_vars = set(x.name for x in tf.global_variables())
            optimizer = tf.train.AdamOptimizer(learning_rate=self.step_size,
                                               name='attack_adam')
            # This term measures the perturbation size.
            # It's supposed to be used when computing minimum perturbation adversarial examples.
            # When performing norm-constrained attacks, self.c_constant should be set to 0.
            dist_term = tf.reduce_sum(
                self.c_constants *
                tf.reduce_sum(tf.square(self.delta), axis=1))
            self.train_step = optimizer.minimize(self.loss + dist_term,
                                                 var_list=[self.delta])
            end_vars = tf.global_variables()
            new_vars = [x for x in end_vars if x.name not in start_vars]
            self.init = tf.variables_initializer(new_vars)
        elif self.optimizer == 'normgrad':
            # Note the minimum pertubation objective is not implemented here
            if self.norm == 'Linf':
                self.train_step = self.delta.assign(
                    self.delta + self.step_size *
                    tf.sign(tf.gradients(-self.loss, self.delta)[0]))
            else:
                grad = tf.gradients(-self.loss, self.delta)[0]
                grad_norm = tf.norm(grad, axis=1, keepdims=True)
                grad_norm = tf.clip_by_value(grad_norm,
                                             np.finfo(float).eps, np.inf)
                self.train_step = self.delta.assign(self.delta +
                                                    self.step_size * grad /
                                                    grad_norm)

        with tf.control_dependencies([self.train_step]):
            # following https://adversarial-ml-tutorial.org/adversarial_examples/
            delta_ = tf.minimum(tf.maximum(self.delta, self.x_min - self.x0),
                                self.x_max - self.x0)
            if self.norm == 'L2':
                norm = tf.norm(delta_, axis=1, keepdims=True)
                # TODO use np.inf instead of tf.reduce_max(norm)
                # delta_ = delta_ * self.max_distance / tf.clip_by_value(norm, clip_value_min=self.max_distance,
                #                                                        clip_value_max=tf.reduce_max(norm))
                bound_norm = tf.clip_by_value(
                    norm,
                    clip_value_min=np.finfo(float).eps,
                    clip_value_max=self.max_distance)
                delta_ = delta_ * bound_norm / tf.clip_by_value(
                    norm,
                    clip_value_min=np.finfo(float).eps,
                    clip_value_max=np.inf)
            else:
                delta_ = tf.clip_by_value(delta_, -self.max_distance,
                                          self.max_distance)
            self.calibrate_delta = self.delta.assign(delta_)

    def perturb(self, x_nat, y, sess, c_constants=None, verbose=False):
        delta = np.zeros_like(x_nat)
        if self.rand:
            if self.norm == 'L2':
                delta = np.random.randn(*x_nat.shape)
                scale = np.random.uniform(low=0.0,
                                          high=self.max_distance,
                                          size=[delta.shape[0], 1])
                delta = scale * delta / np.linalg.norm(
                    delta, axis=1, keepdims=True)
            else:
                delta = np.random.uniform(-self.max_distance,
                                          self.max_distance, x_nat.shape)
            # # This clips (x_nat+delta) to (x_min, x_max), but in practise I found it not neccessary
            # delta = np.minimum(np.maximum(delta, self.x_min - x_nat), self.x_max - x_nat)

        if self.optimizer == 'adam':
            sess.run(self.init)

        if c_constants is None:
            c_constants = np.zeros(x_nat.shape[0])

        if y is None:
            sess.run(
                [self.assign_delta, self.assign_x0, self.assign_c_constants],
                feed_dict={
                    self.delta_input: delta,
                    self.x0_input: x_nat,
                    self.c_constants_input: c_constants
                })
        else:
            sess.run(
                [
                    self.assign_delta, self.assign_x0, self.assign_y,
                    self.assign_c_constants
                ],
                feed_dict={
                    self.delta_input: delta,
                    self.x0_input: x_nat,
                    self.y_input: y,
                    self.c_constants_input: c_constants
                })

        for i in range(self.num_steps):
            sess.run([self.train_step, self.calibrate_delta])

        return sess.run(self.x)

    def batched_perturb(self, x, y, sess):
        adv = []
        for i in range(0, x.shape[0], self.batch_size):
            adv.append(
                self.perturb(x[i:i + self.batch_size],
                             y[i:i + self.batch_size], sess))
        return np.concatenate(adv)


class PGDAttackDetector(PGDAttack):
    def __init__(self, detector, **kwargs):
        super().__init__(**kwargs)
        self.detector_logits = detector.forward(self.x)
        if kwargs['optimizer'] == 'normgrad':
            # normgrad optimizes cross-entropy loss (optimize logit outputs makes no difference)
            labels = tf.zeros_like(self.detector_logits)
            self.loss = -tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=self.detector_logits))
        else:
            self.loss = tf.reduce_sum(-self.detector_logits)
        self.setup_optimizer()




class TwoClassMLP:
    def __init__(self, hidden_layer_sizes, output_size, var_scope):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.var_scope = var_scope

    def forward(self, x):
        with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
            for i, hidden_layer_size in enumerate(self.hidden_layer_sizes):
                x = tf.layers.dense(inputs=x,
                                    name='hidden_{}'.format(i),
                                    units=hidden_layer_size,
                                    activation=tf.nn.relu)

            logits = tf.layers.dense(inputs=x,
                                     name='output',
                                     units=self.output_size)
            if self.output_size == 1:
                logits = tf.squeeze(logits)

            return logits


class Model(object):
    def __init__(self, var_scope, hidden_sizes=[200, 200, 200, 200, 200], input_size=2):
        self.output_size = 1
        self.var_scope = var_scope

        self.y_input = tf.placeholder(tf.int64, shape=[None])

        self.input_size = input_size
        self.x_input = tf.placeholder(tf.float32,
                                      shape=[None, self.input_size])
        self.net = TwoClassMLP(hidden_layer_sizes=hidden_sizes,
                               output_size=1,
                               var_scope=var_scope)
        self.logits = self.net.forward(self.x_input)

        self.y_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(self.y_input, tf.float32), logits=self.logits)

        self.xent = tf.reduce_mean(self.y_xent)

        self.predictions = tf.cast(self.logits > 0, tf.int64)
        
    def forward(self, x):
        return self.net.forward(x)


def plot_detector(detector, x, y, ax, sess, plot_z=True):
    grid_x, grid_y = np.meshgrid(np.arange(0, 1.0, 0.001),
                                 np.arange(0, 1.0, 0.001))
    if plot_z:
        grid_z = sess.run(
            tf.sigmoid(detector.logits),
            feed_dict={detector.x_input:
                       np.c_[grid_x.ravel(), grid_y.ravel()]})
        grid_z = grid_z.reshape(grid_x.shape)
        img = np.zeros([*grid_z.shape, 3])
        img[:, :, 0] = 1-grid_z
        img[:, :, 1] = grid_z
        img[:, :, 2] = 0.0
        ax.imshow(grid_z, extent=(0, 1, 0, 1), origin="lower", alpha=1.0, cmap='inferno')


    if x is not None and y is not None:
        x0, x1 = x[y == 0], x[y == 1]
        ax.scatter(x0[:, 0], x0[:, 1], color='red', marker='.', s=1)
        ax.scatter(x1[:, 0], x1[:, 1], color='blue', marker='.', s=1)
#         ax.axis('equal')
        ax.set_aspect('equal', 'box')
#         ax.scatter(x0[:, 0], x0[:, 1], color='red', marker='.', s=0.3, facecolors='red')
#         ax.scatter(x1[:, 0], x1[:, 1], color='blue', marker='.', s=0.3, facecolors='blue')

#     ax.set_axis_off()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    
def make_dataset(shape_fn, n_pos_samples, n_neg_samples):
    np.random.seed(123)
#     # Make two circles
#     x, y = make_circles(n_samples=n_pos_samples*2, factor=.1, noise=.1)
#     x = (x - x.min())/(x - x.min()).max()
#     x[y==0] = np.random.rand((y==0).sum(), 2)
#     # Replace the inner circle with the grid
#     x[y==1] = shape_fn((y==1).sum())
    
#     x0, y0 = x[y==0], y[y==0]
#     x1, y1 = x[y==1], y[y==1]
    
    x0 = np.random.rand(n_neg_samples*10, 2)
    y0 = np.zeros(x0.shape[0], dtype=np.int64)
    x1 = shape_fn(n_pos_samples)
    y1 = np.ones(x1.shape[0], dtype=np.int64)
    x1_xmin, x1_ymin = np.min(x1, axis=0)
    x1_xmax, x1_ymax = np.max(x1, axis=0)
    xmask = np.logical_and(x0[:,0]>x1_xmin-0.05, x0[:,0]<x1_xmax+0.05)
    ymask = np.logical_and(x0[:,1]>x1_ymin-0.05, x0[:,1]<x1_ymax+0.05)
    mask = np.logical_not(np.logical_and(xmask, ymask))
    x = np.concatenate([x0[mask][:n_neg_samples], x1])
    y = np.concatenate([y0[:n_neg_samples], y1])
    index = np.random.permutation(x.shape[0])
    x, y = x[index], y[index]
    
    return x, y
