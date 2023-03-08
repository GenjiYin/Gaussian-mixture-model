# Gaussian-mixture-model
## 数据形式和相关配置
记
$\mathbf{x}_1$, 
$\mathbf{x}_2$, 
..., 
$\mathbf{x}_n$为观测数据, 且
$z_1$、
$z_2$、
...、
$z_K$ 为类别变量, 这里称其为隐变量. 观测变量服从高斯混合分布: 
$p(x)=\sum\alpha_i N(x; \mu_i, \sigma^2_i)$
其中分布的权重和其中的均值和方法均为需优化的参数. 
## 1. EM算法求解高斯混合模型中的参数
以其中一个样本
$\mathbf{x}_i$
为例
