import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from lbfgs import optimizer as lbfgs_op


class PINNs(models.Model):
    def __init__(self, model, optimizer, epochs, **kwargs):
        super(PINNs, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.hist = []
        self.epoch = 0
        self.sopt = lbfgs_op(self.trainable_variables)
        self.nu = 0.01/np.pi
              
    @tf.function
    def net_f(self, cp):
        """
        function for Burger Equations
        """
        cp = self.scalex_r(cp)
        t = cp[:, 0]
        x = cp[:, 1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(x)
            X = tf.stack([t, x], axis = -1)
            X = self.scalex(X)

            U = self.model(X)
            U = self.scale_r(U)
            U = U[:,0]
            U_t = tape.gradient(U,t)
            U_x = tape.gradient(U,x)

        U_xx = tape.gradient(U_x, x)
        
        """
        u_t + u * u_x - (0.01/pi) * u_xx  = 0
        We want residual to be ZERO
        """
        f = U_t + U * U_x - self.nu* U_xx      
        
        return f

    
    @tf.function
    def train_step(self, bc, cp):
        X = bc[:, :2]
        y = bc[:, 2:]
     
        with tf.GradientTape() as tape:
            u_p_bc = self.model(X)
            
            f = self.net_f(cp)
            
            loss_bc = tf.reduce_mean(tf.square(y - u_p_bc))
           
            loss_f = tf.reduce_mean(tf.square(f))
            
            loss_u = loss_bc
            loss = loss_u + loss_f
            
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_u)
        l3 = tf.reduce_mean(loss_f)
        
        tf.print('loss:', l1, 'loss_u:', l2, 'loss_f:', l3)
        return loss, grads, tf.stack([l1, l2, l3])
    
    
    def fit_scale(self, y):
        ymax = tf.reduce_max(tf.abs(y), axis = 0)
        self.ymax = tf.concat([ymax, [1.0]], 0)
        return y / ymax
    
    @tf.function
    def scale(self, y):
        return y / self.ymax
    
    @tf.function
    def scale_r(self, ys):
        return ys * self.ymax
    
    def fit_scalex(self, x):
        xmax = tf.reduce_max(tf.abs(x), axis = 0)
        xmin = tf.reduce_min(x, axis = 0)
        self.xmax = xmax
        self.xmin = xmin
        xs = ((x - xmin) / (xmax - xmin))
        return xs
    @tf.function
    def scalex(self, x):
        xs = ((x - self.xmin) / (self.xmax - self.xmin)) 
        return xs
    
    @tf.function
    def scalex_r(self, xs):
        x = (xs) * (self.xmax - self.xmin) + self.xmin
        return x
    
    def fit(self, bc, cp):
        bc = tf.convert_to_tensor(bc, tf.float32)
        cp = tf.convert_to_tensor(cp, tf.float32)
        
        x_bc = bc[:, :2]
        y_bc = bc[:, 2:]
        
        y_bc = self.fit_scale(y_bc)
        x_bc = self.fit_scalex(x_bc)
        
        cp = self.scalex(cp)
        bc = tf.concat([x_bc, y_bc], axis = 1)
        
        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(hist.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(hist.numpy())
            
            
            
        self.sopt.minimize(func)
            
        return np.array(self.hist)
    
    def predict(self, cp):
        cp = tf.convert_to_tensor(cp, tf.float32)
        cp = self.scalex(cp)
        u_p = self.model(cp)
        u_p = self.scale_r(u_p)

        return u_p.numpy()