"""
采用EM算法学习混合高斯分布
"""
import tensorflow as tf
from tensorflow.keras import optimizers, Model

class GMM(Model):
    def __init__(self, dim, class_num):
        """
        利用变分推断
        :params dim: 数据维度(维度相互独立)
        :params class_num: 类别数
        """
        self.dim = dim
        self.class_num = class_num

        super(GMM, self).__init__()
        # 初始化均值
        self.u = tf.Variable(tf.random.normal([class_num, dim]))

        # 初始化对数方差
        self.log_var = tf.Variable(tf.random.normal([1, class_num]))

        # 初始化类别的先验分布(广义先验)
        self.log_class_prior = tf.Variable(tf.random.normal([1, class_num]))

    def training(self, data):

        # 初始化优化器
        op = optimizers.Adam()

        # 计算log(p(x|z))
        def log_p_x_z(u, log_var):
            """
            计算p(x|z)的分布矩阵
            :params u: 均值向量
            :params log_var: 对数方差
            """
            u = tf.repeat(tf.reshape(u, [1, -1, self.dim]), len(data), axis=0)
            df = tf.reshape(data, [len(data), 1, self.dim])
            y = df-u
            y = tf.matmul(y, tf.transpose(y, (0, 2, 1)))
            y = tf.linalg.diag_part(y)
            log_p_x_z = -y/(2*tf.exp(log_var))-log_var*self.dim/2
            return log_p_x_z

        for _ in range(4000):
            # 计算p(z|x)
            up = tf.exp(log_p_x_z(self.u, self.log_var) + tf.math.log(tf.nn.softmax(tf.exp(self.log_class_prior))))
            down = tf.matmul(tf.exp(log_p_x_z(self.u, self.log_var)), tf.transpose(tf.nn.softmax(tf.exp(self.log_class_prior)), (1, 0)))
            p_z_x = up/down

            with tf.GradientTape() as tp:
                tp.watch([self.u, self.log_var, self.log_class_prior])
                # 对先验分布正则化
                log_class_prior = tf.math.log(tf.nn.softmax(tf.exp(self.log_class_prior)))

                # 矩阵运算
                log_p_x_z_ = log_p_x_z(self.u, self.log_var)
                y = log_p_x_z_ + log_class_prior
                out = tf.linalg.diag_part(tf.matmul(y, tf.exp(tf.transpose(p_z_x, (1, 0)))))
                out = -tf.reduce_sum(out)
                print(out)
            g = tp.gradient(out, [self.u, self.log_var, self.log_class_prior])
            op.apply_gradients(zip(g, [self.u, self.log_var, self.log_class_prior]))

    def predict(self, data):
        # 先训练后预测
        print("training")
        self.training(data)
        def log_p_x_z(u, log_var):
            """
            计算p(x|z)的分布矩阵
            :params u: 均值向量
            :params log_var: 对数方差
            """
            u = tf.repeat(tf.reshape(u, [1, -1, self.dim]), len(data), axis=0)
            df = tf.reshape(data, [len(data), 1, self.dim])
            y = df-u
            y = tf.matmul(y, tf.transpose(y, (0, 2, 1)))
            y = tf.linalg.diag_part(y)
            log_p_x_z = -y/(2*tf.exp(log_var))-log_var*self.dim/2
            return log_p_x_z
        up = tf.exp(log_p_x_z(self.u, self.log_var) + tf.math.log(tf.nn.softmax(tf.exp(self.log_class_prior))))
        down = tf.matmul(tf.exp(log_p_x_z(self.u, self.log_var)), tf.transpose(tf.nn.softmax(tf.exp(self.log_class_prior)), (1, 0)))
        p_z_x = up/down
        return tf.argmax(p_z_x, axis=1)



if __name__ == "__main__":
    # a = tf.constant([[1.0, 2.0], [3.0, 2.0], [3.1, 2.5], [10, 8]])
    # m = GMM(2, 2)

    # print(m.predict(a))

    # 导入数据测试
    import pandas as pd
    import numpy as np

    data = pd.read_excel('./1.xlsx')
    data = np.array(data)
    data = tf.constant(data, dtype=tf.float32)

    m = GMM(2, 3)
    print(m.predict(data))




