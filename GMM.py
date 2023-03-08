"""
采用变分推断的方式无监督学习混合高斯分布
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
        # 初始化隐变量近似分布(未正则化)
        self.log_q_z_x = tf.Variable(tf.random.normal([len(data), self.class_num]))

        # 初始化优化器
        op = optimizers.Adam()

        for _ in range(3500):
            # EM算法需要上一步的参数需要在梯度记录外赋值, 但此处是变分推断
            with tf.GradientTape() as tp:
                tp.watch([self.u, self.log_var, self.log_class_prior, self.log_q_z_x])
                # 对先验分布正则化
                log_class_prior = tf.math.log(tf.nn.softmax(tf.exp(self.log_class_prior)))

                # 对隐变量近似分布正则化
                q_z_x = tf.nn.softmax(tf.exp(self.log_q_z_x))

                # 矩阵运算
                u = tf.repeat(tf.reshape(self.u, [1, -1, self.dim]), len(data), axis=0)
                df = tf.reshape(data, [len(data), 1, self.dim])
                y = df-u
                y = tf.matmul(y, tf.transpose(y, (0, 2, 1)))
                y = tf.linalg.diag_part(y)
                log_p_x_z = -y/(2*tf.exp(self.log_var))-self.log_var*self.dim/2
                y = log_p_x_z + log_class_prior
                out = tf.linalg.diag_part(tf.matmul(y, tf.exp(tf.transpose(q_z_x, (1, 0)))))
                out = -tf.reduce_sum(out)
                print(out)
            g = tp.gradient(out, [self.u, self.log_var, self.log_class_prior, self.log_q_z_x])
            op.apply_gradients(zip(g, [self.u, self.log_var, self.log_class_prior, self.log_q_z_x]))

    def predict(self, data):
        # 先训练后预测
        print("training")
        self.training(data)
        p = tf.exp(self.log_q_z_x)
        p = tf.nn.softmax(p, axis=1)
        return tf.argmax(p, axis=1)




if __name__ == "__main__":
    a = tf.random.normal([4, 2])
    b = tf.matmul(a, tf.transpose(a, [1, 0]))

    a = tf.constant([[1.0, 2.0], [3.0, 2.0], [3.1, 2.5], [6, 8]])
    m = GMM(2, 2)
    m.training(a)
    print(m.predict(a))

