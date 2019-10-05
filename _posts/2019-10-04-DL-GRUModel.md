---
layout: post
title:  "深度学习之-GRU模型"
date:   2019-10-04 14:32:01
categories: 深度学习
tags: 深度学习  GRU
excerpt: 记录一下关于GRU网络的内容
---


* content
{:toc}
由于这几天正在看阿里发表的CRT模型 DIEN的过程中遇到了一个新的问题，不懂GRU模型。而这又是这篇模型的关键，没办法只能自己慢慢补好了。接下来就简单记录一下自己看的深度学习模型---GRU

## 1、GRU前提
由于GRU(Gated Recurrent Unit)是LSTM的改进，所以还是简单的复习一下LSTM吧。

#### LSTM
LSTM是一种特殊的RNN网络，也是比较好的解决RNN所出现的梯度消失(爆炸)以及不能学习到长间隔词语问题的一个网络。
![LSTM结构](https://img-blog.csdn.net/20170919123006918?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHJlYWRlcmw=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

LSTM的主要思想是‘细胞状态’，细胞状态其实指的就是上图中最上面的那一个类似于传送带的结构。通过细胞状态，可以使得信息可以在上面流传，实现可以学习到长间隔语句的信息。

LSTM通过设计4个叫做‘门’的结构，对细胞状态中加入或减少信息。门就是通过sigmoid和pointwise乘法，选择性的让信息通过的方式。其中sigmoid输出的范围在0-1之间，表示的就是每部分通过的量。

![](https://img-blog.csdn.net/20170919125017539?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHJlYWRlcmw=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


接下来开始介绍LSTM的四个步奏
第一步就是遗忘门，如下图。遗忘门的主要作用就是就是用来决定从细胞状态中丢去什么信息。其输入是当前输入x和上一个细胞的状态$h_{t-1}$，通过sigmoid函数得到决定丢弃的信息。这一个步的函数表示为

$$f_t = \sigma (W_f[h_{t-1},x_t]+b)$$


![](https://upload-images.jianshu.io/upload_images/3426235-671c44df2719ee97.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这一个步的函数表示为
$$f_t = \sigma(W_f[h_{t-1},x_t]+b)$$

第二步进行的就是输入门，如下图。输入门包括两部份组成,一部分是sigmoid函数，其输出的$i_t$决定着我们要更新的值，第二部分是tanh函数，这部分输出的结果$a_t$是一个新的候选值，并通过和$i_t$相乘决定将多少加入到细胞状态中。

![](https://upload-images.jianshu.io/upload_images/3426235-2a78bf02e451fa6b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这一个步的函数表示为
$$i_t = \sigma(W_i[h_{t-1},x_t]+b)$$
$$a_t = tanh(W_C[h_{t-1},x_t]+b)$$

第三步就是更新细胞状态，将细胞状态从$C_{t-1}$更新到$C_t$状态。这一过程主要是两步，先与遗忘门的输出$f_t$相乘，目的是将就细胞状态中的一些内容进行遗忘。接着就是和 it * ~Ct进行相加，这是为了在细胞状态中加入通过输入门得出新的候选值向量，在细胞中更新新的信息。


![](https://upload-images.jianshu.io/upload_images/3426235-aca1efb0a7eb189e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这一部的函数表示为

$$C_t = f_t * C_{t-1} + i_t*a_t$$

第四步就是确定输出了,也叫做输出门。这一步也通过细胞状态进行决定输出的内容。首先这里会先进性一个sigmoid函数，主要是选择出要输出的这一部分，然后就是通过一个tanh函数进行处理将细胞状态映射在-1-1的范围内，接着和sigmoid函数的输出相乘，得到要输出的那一部分内容。


 ![](https://upload-images.jianshu.io/upload_images/3426235-45af9b3908e1c3ad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这一步的函数表示为
$$o_t = \sigma(W_o[h_{t-1},x_t]+b)$$
$$h_t =o_t*tanh(C_{t})$$


## 2、GRU(Gated Recurrent Unit)
#### GRU结构
上面就简单的介绍完了LSTM的基本结构了，可以知道的是LSTM存在着三个门 "遗忘门"、"输入门"、"输出门"。在GRU中将LSTM中的输入门和遗忘门进行融合称为一个门称为"重置门",另一个叫"更新门"。即图中的$z_t$和$r_t$.其中$r_t$是重置门，通过当前输入和上一个隐藏层状态，通过sigmoid函得到要将上一个状态的信息带入新细胞的程度，这个值越大表示保留之前状态越多，反之保留的越少。重置门$r_t$这是控制对上一个状态的遗忘程度，值越大表示遗忘的越多，反之遗忘的越少。

![](https://upload-images.jianshu.io/upload_images/3426235-344f0cb517558041.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 具体介绍
上面总的说了一下门的含义，下面具体说一下每个门的含义和公式表达。

##### 1.更新门
更新门是当$x_t$插入网络单元时，它乘以自己的权重W（z）。对于$h_（t-1）$也是一样，它保存了先前$t-1$个单位的信息，并乘以它自己的权重U（z）。将两个结果加在一起，然后应用S型激活函数将结果压缩在0和1之间。所以更新门可帮助模型确定需要将多少过去信息（来自先前的时间步长）传递给未来。这确实非常强大，因为该模型可以决定复制过去的所有信息，并消除了梯度问题消失的风险。

下图是从大佬那里拿来的，感觉画的太好了，哈哈哈哈！


![](https://upload-images.jianshu.io/upload_images/3426235-15057ec61bcdd499.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

看到图中的更新门，可以得到公式如下：
$$z^{(t) = \sigma(W^{(z)}x^{(t)}+U^{(z)}h^{(t-1)})}$$

##### 2.重置门
重置门从模型中使用此门来决定要忘记多少过去的信息。

下图是从大佬那里拿来的，感觉画的太好了，哈哈哈哈！
![](https://upload-images.jianshu.io/upload_images/3426235-15057ec61bcdd499.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

看到图中的重置门，可以得到公式如下：
$$z^{(t) = \sigma(W^{(r)}x^{(t)}+U^{(r)}h^{(t-1)})}$$

#### 3.当前内存内容
首先，我们从使用复位门开始。我们引入了一个新的存储器内容，它将使用复位门来存储过去的相关信息。首先输入x_t乘以权重W，h_（t-1）乘以权重U。其次计算重置门r_t和Uh_（t-1）之间的Hadamard（逐元）乘积。这将决定从先前的时间步骤中删除什么。最后应用非线性激活函数tanh。
计算公式如下：
$$h_t^' = tanh(Wx^{(t)}+r_t \cdot Uh^{(t-1)})$$

#### 4.当前时间步的最终记忆
最后，网络需要计算h_t-向量，该向量保存当前单位的信息并将其传递给网络。为此，需要更新门。它确定从当前内存内容中收集什么$h'_t$ 以及从先前步骤中收集的内容-h_ （t-1）。计算公式如下：

$$h_t=z_t \cdot h_{t-1} +(1-z_t) \cdot h_t^{'}$$

对更新门z_t和h_（t-1）应用逐元素乘法。
对（1-z_t）和h'_t应用逐元素乘法。
对步骤1和2的结果求和。

以上就是GRU的大致结构了，具体的参数迭代学习就不多说了，复杂的数学公式我也很难懂。哈哈哈哈，所以继续去看我的DIEN咯！！


### 引用
下面给出参考的一些大神的博客

[GRU和LSTM总结](https://blog.csdn.net/lreaderl/article/details/78022724)

[图解LSTM与GRU单元的各个公式和区别](http://www.sohu.com/a/336551522_99979179)

[Understanding GRU Networks](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)










