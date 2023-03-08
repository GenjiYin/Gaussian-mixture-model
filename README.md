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
网上有很多教程, 这里略去. 

## 2. 变分推断
以其中一个样本为例, 变换其对数似然: 

$\log\{p(x_i)\}=\log\{\frac{p(\mathbf{x}_i, z_j)}{p(z_j|\mathbf{x}_i)}\}$

直接求解
$p(z_j|\mathbf{x}_i)$过于复杂, 这里采用变分推断的方法, 用
$q_i(z_j)$
去逼近该后验分布, 代码中具体操作流程是初始化一个n行K列的
$[\log\{q_i(z_j)\}]$
矩阵. 

对对数似然进一步作变换: 

$\log\{\frac{p(\mathbf{x}_i, z_j)}{p(z_j|\mathbf{x}_i)}\}=\log\{\frac{p(\mathbf{x}_i, z_j)}{q_i(z_j)}\}+\log\{\frac{q_i(z_j)}{p(z_j|\mathbf{x}_i)}\}$

进一步利用近似分布求期望可得: 

$\log\{p(x_i)\}=E_{q}(\log\{\frac{p(\mathbf{x}_i, z_j)}{q_i(z_j)}\})+KL(q||p(z_j|\mathbf{x}_i))$

第二项为KL散度. 最小化KL散度只需最大化第一项即可. 对所有参数采用梯度下降即可得到迭代解.  
