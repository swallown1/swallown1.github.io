---
layout: post
title:  "推荐系统之-DeepFM模型"
date:   2019-09-30 10:31:16
categories: 推荐系统
tags: 推荐系统  DeepFM
excerpt: 记录看过的一些paper
---


* content
{:toc}
最近想看一些关于推荐系统的内容，打算从CRT模型整体的发展过程看一些经典的模型。本篇就是其中一篇，PNN模型，全称Product-based Neural Network。

## 1、原理和模型结构
这篇文章提出的主要问题是，在embedding后到神经网络中得到的交叉特征表达的不够充分，因此提出了一种 product-based 的思想，通过乘法的运算来体现出特征交叉的网络结构。具体的结构如下图：
<img src="https://upload-images.jianshu.io/upload_images/4155986-9867a7134749f48e.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" />

从上往下一次介绍模型的每个结构：

#### 输出层
输出层主要是将上一层的结果拿来做了一个全连接，然后通过sigmoid函数，将结果映射在(0,1)上，得到需要的点击率。
**$$y^{hat} = \sigma(W_3l_2+b_3)$$**

#### l2层
l2层主要是将上一层的结果拿来做了一个全连接，然后通过relu函数，得到l2层的输出结果。
**$$l_2 = relu(W_2l_1+b_2)$$**

#### l1层
l1层和l2层类似，公式如下。
**$$l_1= relu(l_z+l_p+b_1)$$**
这里的$l_z$和$l_p$ 是什么呢？明显不是embedding层的结果啊。其实这就是PNN全文的核心，在embedding层和MLP中间加入的product layer.

#### Product layer
Product layer 其核心思想就是想体现一种多个特征之间更多的是一种“且”的关系，而不是和的关系。例如，性别为男且喜欢游戏的人群，比起性别男和喜欢游戏的人群，前者的组合比后者更能体现特征交叉的意义。

Product layer包含着2部分，其中的$l_z$和$l_p$，所表达的是线性和非线性关系。形式如下：
<img src='https://upload-images.jianshu.io/upload_images/4155986-79596b0e03993e0d.png?imageMogr2/auto-orient/strip|imageView2/2/w/704/format/webp' />
上面的运算方式指的是矩阵的点乘：
<img src='https://upload-images.jianshu.io/upload_images/4155986-cc22b83064d309cf.png?imageMogr2/auto-orient/strip|imageView2/2/w/378/format/webp'>

#### Embedding 层
这里的Embedding层和DeepFM中的是类似的，对于不同field的特征转化成相同长度的向量，这里用f来表示.
### $$(f_1,f_2,...,f_n)$$

#### 损失函数
损失函数采用的是和逻辑回归相同的函数，公式如下：
### $$Loss(y,y^{hat}) = -y*logy^{hat}-(1-y)*log(1-y^{hat})$$

## 2、Product Layer 详细解释
前面说，product layer有两部分，$l_z$和$l_p$,其中这两部分都是由Embedding得到的。$l_z$是表示线性向量，因此我们直接由Embedding得到,z就是embedding层的复制。
#### $$Z = (z_1,z_2,...,z_n) = (f_1,f_2,...,f_n)$$

对于p来说，需要将Embedding映射成另一个向量
$$p=\{ p_{ij}\} , i,j=1,2,...,n$$
$$p_{i,j} = g(f_i,f_j)$$

其中这个g函数就可以是不同的类型了。文中给出了两种函数，一种是Inner PNN，另一种是 Outer PNN 

#### 2.1、IPNN
先定义Embedding的大小为M，field的大小为N，而lz和lp的长度为D1。
IPNN结构如下图所示：
<img src='https://upload-images.jianshu.io/upload_images/4155986-efc8f371d4e694a4.png?imageMogr2/auto-orient/strip|imageView2/2/w/740/format/webp' style="height:50%;width:50%;"/>

在IPNN模型中，g函数采用的就是内积来代表pij：
$$g(f_i,f_j)=<f_i,f_j>$$

从中我们可以看出，p是一个对称矩阵，因此W也可以是一个对称矩阵，就有如下分解：
#### $$W_p^n = \theta^n\theta^{nT}$$
因此：
#### $$l_p^n = W_p^n.p =\sum_{i=1}^n\sum_{j=1}^n\theta^n\theta^{nT}<f_i,f_j>=<\sum_{i=1}^N \delta_i^n,\sum_{i=1}^N\delta_i^n> $$
#### $$\delta_i^n=\theta^n_if_i$$
因此：
#### $$\delta^n = (\delta_1^n,\delta_2^n,...,\delta_n^n)$$
从而得到：
#### $$l_p=(||\sum_i\delta^1_i||,||\sum_i\delta^2_i||,...,||\sum_i\delta^{D1}_i||)$$

#### 2.2、OPINN
OPNN的示意图如下：
<img src='https://upload-images.jianshu.io/upload_images/4155986-d9924e3ef896dc31.png?imageMogr2/auto-orient/strip|imageView2/2/w/756/format/webp' style="height:50%;width:50%;"/>

在OPNN中，g函数的计算方式如下：
$$p_{i,j}=g(f_i,f_j)=f_if_j^T$$
这里的$p_{i,j}$是一个M* M的矩阵，由于计算Pi,j的复杂度是o(M* M),因此计算p的时间复杂度是 N* N*M *M,可以看出这个计算的代价很大。文中对此进行了优化，采用了叠加的方式，重新定义p
矩阵
$$p=\sum^N_{i=1}\sum^N_{j=1}f_if^T_j=f\sum(f\sum)^T ,f\sum = \sum^N_{i=1} f_i$$
此时时间复杂度D1 * M *(M+N)









