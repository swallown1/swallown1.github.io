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
这篇主要讲的是一个关于CRT的一个深度学习模型 DeepFM。DeepFM算法主要包含两个方面，其中是FM矩阵分解，其次是MLP，将两者结合起来的模型。其中FM矩阵分解主要是将提取一阶和二阶的低阶特征进行提取。MLP主要是提取FM无法提取的高阶特征。

## 1、原理和背景
这篇文章提出的主要问题是，FM虽然理论上可以进行高阶的特征提取，但是因为计算的复杂性的问题，现实中只进行二阶特征的提取。所以对于更高阶的特征组合就无法进行提取了，因此希望通过DNN网络可以对更高阶的特征进行提取。

##### DNN模型

在CRT问题中，由于存在大量的稀疏特征矩阵，因此通过全连接的DNN网络会产生很大的参数，同时很多参数由于都是稀疏的，因此导致反向传播时会出现梯度消失的情况。
<img src='https://upload-images.jianshu.io/upload_images/4155986-f4363ca2be689dbb.png?imageMogr2/auto-orient/strip|imageView2/2/w/1164/format/webp'/>
所以论文中对此的处理是和FFM中的思想类似，将不同的特征分成不同的field
<img src="https://upload-images.jianshu.io/upload_images/4155986-5f476d2c5b616232.png?imageMogr2/auto-orient/strip|imageView2/2/w/1101/format/webp" />

进行处理后，将得到的Dense Vector输入到DNN当中，但是学到的低阶和高阶的特征组合隐藏在MLP当中，但是如何将低阶的特征组合单独建模然后将其和高阶特征组合融合呢?
<img src='https://upload-images.jianshu.io/upload_images/4155986-7e036f56982d323b.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp'/>

那么就是要将DNN和FM进行融合才能达到这个要求了
<img src='https://upload-images.jianshu.io/upload_images/4155986-2b8d2e22017ad339.png?imageMogr2/auto-orient/strip|imageView2/2/w/1183/format/webp'/>

这里就存在两种方式，一种是两者进行串联 ，另一种是进行并联。DeepFMM模型就是属于并行模型。



## 2、DeepFM模型
如下就是DeepFM的模型：
<img src='https://upload-images.jianshu.io/upload_images/4155986-21fa429e42108e99.png?imageMogr2/auto-orient/strip|imageView2/2/w/535/format/webp' />

DeepFM模型中包含两部分：DNN和因子分解机(FM)，这两者分别负责高阶特征组合和低阶特征组合。两者有相同的输入，DeepFM模型的公式如下：
## $$y^{hat} =sigmoid(y_{FM}+y_{DNN})$$

对于FM部分，是一个因子分解机，其输出公式如下：
#### $$y_{FM}=\sum_i W_ix_i + \sum_{i=1}^d\sum_{j=1}^d<v_i,v_j>x_i*x_j$$

对于深度模型部分，对于一般问题，直接将输入值输入到DNN中，但是CRT问题中绝大多数的输入都是稀疏矩阵，因此无法直接输入。因此在进入深度模型的过程中一般会进行Embedding层，将稀疏向量转化成Dense 向量，将高维的稀疏向量转化成低维的稠密向量。

