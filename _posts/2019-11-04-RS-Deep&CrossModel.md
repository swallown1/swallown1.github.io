---
layout: post
title:  "推荐系统之-Deep&Cross模型"
date:   2019-11-04 13:28:45
categories: 推荐系统
tags: 推荐系统  Deep&Cross
excerpt: Deep & Cross Network(DCN)[1]是来自于 2017 年 google 和 Stanford 共同完成的一篇工作
---


* content
{:toc}
Deep & Cross Network(DCN)[1]是来自于 2017 年 google 和 Stanford 共同完成的一篇工作。这篇文中可以说在Wide & Deep模型的基础上，将其中的wide部分改成了Cross部分。相比于DeepFM模型来说，Cross代替FM部分更好地拥有高的计算效率并且能够提取到更高阶的交叉特征。

### DCN原理
在之前说过的Wide&Deep中，我们知道模型的wide部分是交叉组合特征依然需要依靠hand-craft来完成；在DeepFM中，虽然不需要用人工的方式进行特征交叉，但是这也仅限于二阶交叉。
本文提出了一种交叉的网络结构的深度学习模型DCN，Deep & Cross Network，能对sparse和dense的输入自动学习特征交叉，可以有效地捕获有限阶（bounded degrees）上的有效特征交叉，无需人工特征工程或暴力搜索（exhaustive searching），并且计算代价较低。，在CTR预估方面可以取得较好的效果。
我们把这么模型简称为DCN。

DCN模型主要包含4个部分

1.Embedding and Stacking Layer 对稀疏输出入的处理

2.Cross Network 进行显式特征交叉

3.Deep Network 捕获高阶的特征

4.Combination Layer 进行两部分的特征结合


接下来仔细讲一下这几个部分(模型整体结构图如下)
<img src="https://swallown1.github.io/image/DCN.png" />


### Embedding and Stacking Layer
这里的Embedding和其他模型的Embedding类似，也是用来处理CRT中稀疏的问题，但是不同的是，该模型考虑的是具有离散和连续两种特征的输入数据。对于离散特征，通常都是通过进行onehot进行处理，问题在于会出现高维度的特征空间，为了减少维数，我们采用嵌入的方式将这些离散特征转化成稠密的特征(通常称为嵌入向量)

$$X_{embed,i} = W_{embed,i}X_i$$

然后将得到的嵌入向量与连续特征向量拼接起来，

$$x_0 = [ x_{embed,1}^T, ..., X_{embed,k}^T, X_{dense}^T]。$$

称为X0作为Cross Network和Deep Network的输入。
```
# model
self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
self.embeddings = tf.multiply(self.embeddings,feat_value)

self.x0 = tf.concat([self.numeric_value,
                     tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])]
                    ,axis=1)
```

### Cross Network
交叉网络是这篇文章中的一个创新部分，主要是用来进行计算组合特征。

交叉网络的核心就是通过显示特征交叉。交叉网络是由交叉层组成的，通过前一层的结果和初始值x0进行交叉，在乘以权重得到高一阶的特征，
再加上前一层的结果，得到高一阶的特征交叉。每层具有以下的公式

$$x_{l+1}= x_0x_l^Tw_1+b_l+w_l=f(x_l,w_1,b_1)+x_l$$

其中：
xl,xl+1是列向量（column vectors），分别表示来自第l层和第(l+1)层cross layers的输出；
wl,bl∈Rd是第l层layer的weight和bias参数。

交叉层的可视图如下所示：
<img src="https://swallown1.github.io/image/cross_layer.webp" />

通过图示可以看出，交叉网络特殊的结构使得交叉特征程度随着层数深度的增加而增大。多项式的最高程度（就输入X0而言）为L层交叉网络L + 1。如果用Lc表示交叉层数，d表示输入维度。然后，参数的数量参与跨网络参数为：d Lc 2 (w和b).
因为每一层的W和b都是d维度的。从上式可以发现，复杂度是输入维度d的线性函数。所以相比于deep network，cross network引入的复杂度微不足道。这样就保证了DCN的复杂度和DNN是一个级别的。论文中表示，Cross Network之所以能够高效的学习组合特征，就是因为x0 * xT的秩为1( rank-one 特性(两个向量的叉积))，使得我们不用计算并存储整个的矩阵就可以得到所有的cross terms
```
# cross_part
self._x0 = tf.reshape(self.x0, (-1, self.total_size, 1))
x_l = self._x0
for l in range(self.cross_layer_num):
    x_l = tf.tensordot(tf.matmul(self._x0, x_l, transpose_b=True),
                        self.weights["cross_layer_%d" % l],1) + self.weights["cross_bias_%d" % l] + x_l
#变成列向量
self.cross_network_out = tf.reshape(x_l, (-1, self.total_size))
```

### Deep Network
这部分就是通过MLP进行高阶特征的提取，进行一个全连接的前反馈神经网络，公式如下：
$$h_{l+1}=f(W_1h_1+b_1)$$

```
self.y_deep = tf.nn.dropout(self.x0,self.dropout_keep_deep[0])

for i in range(0,len(self.deep_layers)):
    self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["deep_layer_%d" %i]), self.weights["deep_bias_%d"%I])
    self.y_deep = self.deep_layers_activation(self.y_deep)
    self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])
```

### Combination Layer
将Cross layer和Deep layer两个并行网络出来的输出做一次concat,链接层将两个并行网络的输出连接起来，经过一层激活函数得到输出：

$$p = \sigma \left( \left[ \mathbf { x } _ { L _ { 1 } } ^ { T } , \mathbf { h } _ { L _ { 2 } } ^ { T } \right] \mathbf { w } _ { \operatorname { logits } } \right)$$

对于二分类问题这里使用对数损失函数，进行损失计算，形式如下

$$loss = - frac{1}{N} \sum^N_{i=1}y_i log(p_i)+(1-y_i)log(1-p_i)+\lambda \sum_l ||w_l||^2$$

```
# concat_part
concat_input = tf.concat([self.cross_network_out, self.y_deep], axis=1)

self.out = tf.add(tf.matmul(concat_input,self.weights['concat_projection']),self.weights['concat_bias'])


# loss
if self.loss_type == "logloss":
    self.out = tf.nn.sigmoid(self.out)
    self.loss = tf.losses.log_loss(self.label, self.out)
elif self.loss_type == "mse":
    self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
# l2 regularization on weights
if self.l2_reg > 0:
    self.loss += tf.contrib.layers.l2_regularizer(
        self.l2_reg)(self.weights["concat_projection"])
    for i in range(len(self.deep_layers)):
        self.loss += tf.contrib.layers.l2_regularizer(
            self.l2_reg)(self.weights["deep_layer_%d" % I])
    for i in range(self.cross_layer_num):
        self.loss += tf.contrib.layers.l2_regularizer(
            self.l2_reg)(self.weights["cross_layer_%d" % I])
            
```


论文最重要的部分：

1.提出了一种新的交叉网络，在每个层上明确地应用特征交叉，有效地学习有界度的预测交叉特征，并且不需要手工特征工程或穷举搜索。

2.交叉网络（DCN）在LogLoss上与DNN相比少了近一个量级的参数量，所以模型更小。


### References:
1、[推荐系统遇上深度学习(五)--Deep&Cross Network模型理](https://www.jianshu.com/p/77719fc252fa)

2、[推荐系统CTR实战——Deep & Cross](https://fuhailin.github.io/Deep-and-Cross-Network/)

3、[论文](https://arxiv.org/abs/1708.05123)

