import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def linear(x_, fan_in, fan_out):
    if get_dims(x_) == 1:
        x_ = tf.reshape(x_, [-1, 1])
    w = tf.get_variable(name="weights",
                        shape=[fan_in, fan_out],
                        initializer= tf.random_normal_initializer())
    b = tf.get_variable(name="biases",
                        shape=[fan_out],
                        initializer= tf.random_normal_initializer())
    return tf.matmul(x_, w) + b

def get_shape(x_):
    return x_.get_shape().as_list()

def get_dims(x_):
    return x_.get_shape().ndims

def gaussian_function_single(x, mu=0 ,sigma=1):
    return math.exp(-1*(x-mu)*(x-mu)/2/sigma/sigma)/sigma/math.sqrt(2*math.pi)
        
def gaussian_function(x, mu=0 ,sigma=1):
    if type(x) == list:
        x = np.array(x)
    if type(x).__module__==np.__name__:
        a = np.zeros(len(x))
        for i in range(len(x)):
            a[i] = gaussian_function_single(x[i],mu,sigma)
        return a
    else :
        return gaussian_function_single(x, mu, sigma)
def print_keys(string):
    print("Collection name : {}".format(string))
    i = 0
    while True:
        try:
            print(tf.get_collection(string)[i])
            i+=1
        except IndexError:
            break;
def sampling(nsamples, pd):
    return pd['sigma']*np.random.rand(nsamples) + pd['mu']

def pd_spec(pd):
    return "mu = {}, sigma = {}".format(pd['mu'], pd['sigma'])

def clip_op(clip, string):
    i = 0
    while True:
        try:
            temp = tf.global_variables()[i]
            if temp.op.name == "step":
                step = temp
            i+=1
        except IndexError:
            step = tf.Variable(0.0, name = "step")
            break

    count = tf.assign(step, step + 1)
    i = 0
    while True:
        try:
            temp = tf.get_collection(string)[i]
            if temp.op.name.endswith('weights'):
                new_temp = tf.clip_by_value(tf.get_collection(string)[i]
                                        , clip_value_max = clip
                                        , clip_value_min = -clip)
                count = tf.group(count, tf.assign(temp, new_temp))
            i+=1
        except IndexError:
            break;
    return count
