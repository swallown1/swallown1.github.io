---
layout: post
title:  "推荐系统之-Deep&Cross模型"
date:   2019-09-30 10:31:16
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
<img src="https://swallown1.github.io/image/cross_layer.webp" />
