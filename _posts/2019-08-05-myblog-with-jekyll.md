---
layout: post
title:  "Jekyll＋Github 搭建博客"
date:   2019-08-05 19:15:24
categories: jekyll
tags: jekyll github 
excerpt: 记录搭博客的经过
---


* content
{:toc}

一直以来项搭建一个自己的博客，之前用wordpress搭过一个，不过后来一直没用了
(总是做到一半就没坚持下去)。无意之间看到知乎上的相关内容，刚好想搭个博客让自己存放自己的学习笔记，所以希望自己能一直把他用下去吧！

下面简单介绍一下自己搭博客的历程吧，对于不需要域名，只需要本地搭建的就可以拥有一个，你还在等什么？


## 搭建过程

在jekyll的官网上 [http://jekyllrb.com/](http://jekyllrb.com/) 
其实已经说得比较明白了，我在这里还是简单的说一下吧。因为在实验室用的是liunx系统，所以就以liunx为例。    
主要环节有：安装Ruby，安装jekyll，安装代码高亮插件
### 安装Ruby

ruby官网下载安装：[https://www.ruby-lang.org/en/downloads/](https://www.ruby-lang.org/en/downloads/)

还可以用命令安装ruby和rubygems:

>sudo apt-get install rubygems ruby1.9.1-dev

添加path

>export PATH=$PATH:$HOME/bin:/var/lib/gems/1.8/bin

切换成淘宝镜像(有梯子的可以自动略过)
>$ sudo gem sources --remove http://rubygems.org/

>$ sudo gem sources -a http://ruby.taobao.org/

可以使用sudo gem sources -l查看当前设置的服务器。

### 安装jekyll
直接上命令
>sudo gem install jekyll 

至此jekyll就已经安装完毕了，后续就是个性化的自己设定了。

### 创建博客
jekyll官方建立了一个页面，里面有许多的模板可供参考。接下来我们就要奉行“拿来主义”了，将别人的模板为我们所用。

我自己用了徐代龙的技术专栏 的[Jekyll Bootstrap](https://github.com/643435675/643435675.github.io)的模板。下面假设你已经安装了git，我们把他人的网站代码clone下来，为了举例方便，还是选取了Yukang’s Page：
>git clone https://github.com/643435675/643435675.github.io.git

进入文件运行拿来的模板

>jekyll s

突然这时候就出现了一个错误
```
jekyll serve
Configuration file: c:/gitWorkSpace/blog-based-on-jekyll-3/_config.yml
  Dependency Error: Yikes! It looks like you don't have jekyll-paginate or one of its dependencies installed. In order to use Jekyll as currently configured, you'll need to install this gem. The full error message from Ruby is: 'cannot load such file -- jekyll-paginate' If you run into trouble, you can find helpful resources at http://jekyllrb.com/help/!
jekyll 3.1.2 | Error:  jekyll-paginate
```
### 解决方法

我们安装这个插件到本地即可。

```
gem install jekyll-paginate
Fetching: jekyll-paginate-1.1.0.gem (100%)
Successfully installed jekyll-paginate-1.1.0
Parsing documentation for jekyll-paginate-1.1.0
Installing ri documentation for jekyll-paginate-1.1.0
Done installing documentation for jekyll-paginate after 3 seconds
1 gem installed
```

当然就可以启动jekyll服务了，如下：

```
jekyll serve 
```

这样在浏览器中输入 http://localhost:4001/ 就可以访问了。

## Github Pages
前面已经搭建好本地博客了，改完之后怎么将本地的发布到网上呢？当然可以通过github啦，神奇

到 [Github](https://github.com)的官方网站 注册账户，记住自己的用户名，后面会常用到。

>安装git [直接看大佬的吧](https://www.liaoxuefeng.com/wiki/896043488029600)

安装完成在命令行里设置你的git用户名和邮箱：
```
$ git config --global user.name "{username}"          // 用你的用户名替换{username}
$ git config --global user.email "{name@site.com}"    // 用你的邮箱替换{name@site.com}
```

SSH配置
为了和Github的远程仓库进行传输，需要进行SSH加密设置。

在刚才打开的Shell内执行：
```
$ ssh-keygen -t rsa -C"{name@site.com}"    // 用你的邮箱替换{name@site.com}
```
可以不输入其他信息，一直敲回车直到命令完成。
这时你的用户目录（win7以上系统默认在 C:\Users\你的计算机用户名）内会出现名为 .ssh 的文件夹，点进去能看到 id_rsa 和 id_rsa.pub 两个文件，其中 id_rsa 是私钥，不能让怪人拿走， id_rsa.pub 是公钥，无需保密（原理请自行参看密码学.............................................我相信你也不会看）。
接下来用你的浏览器登录Github，点击右上角的“Settings”：
![１](./2015-02-12-create-my-blog-with-jekyll_images/f8afaa39.png)
用文字处理软件打开刚才的 id_rsa.pub 文件，复制全部内容。
点击“SSH Keys”，“Add SSH Key”，将复制的内容粘贴在Key中，点“Add Key”确定。
![２](./2015-02-12-create-my-blog-with-jekyll_images/d7e25742.png)
至此SSH配置完毕。

## 同步仓库
先创建一个仓库(如何创建仓库呢？　[直接看这吧](http://mcace.me/github-pages/jekyll/2018/06/17/use-github-pages.html))，然后就是将刚才本地运行的博客文件上传到github上，

先删除.git文件
```
rm -rf .git
```
然后提交到你创建的仓库
```
git init
git add -A
git commit -m "first commit"
git remote add origin https://github.com/USERNAME/USERNAME.github.io.git
git push -u origin master
```

每当你对本地仓库里的文件进行了修改，只需依次执行以下三个命令即可将修改同步到Github，刷新网站页面就能看到修改后的网页：
```
$ git add .
$ git commit -m "statement"   //此处statement填写此次提交修改的内容，作为日后查阅
$ git push origin master
```


## 后续

*  整个安装过程参考了jekyll官网，注意jekyll还有一个简体中文官网，不过比较坑（我就被坑了），有些内容没有翻译过来，有可能会走弯路，建议如果想看中文的相关资料，也要中英对照着阅读。[jekyll中文网 http://jekyllcn.com](http://jekyllcn.com), [jekyll英文网 http://jekyllrb.com](http://jekyllrb.com)
*  jekyll中的css是用sass写的，当然直接在`_sass/_layout.scss`中添加css也是可以的。
*  本文是用Markdown格式来写的，相关语法可参考： [Markdown 语法说明 (简体中文版) http://wowubuntu.com/markdown/](http://wowubuntu.com/markdown/)  
*  按照本文的说明搭建完博客后，用`github Pages`托管就可以看到了。注意，在github上面好像不支持rouge，所以要push到github上时，我将配置文件_config.yml中的代码高亮改变为`highlighter: pygments`就可以了
*  博客默认是没有评论系统的，本文的评论系统使用了多说，详细安装办法可访问[多说官网 http://duoshuo.com/](http://duoshuo.com/)，当然也可以使用[搜狐畅言 http://changyan.sohu.com/](http://changyan.sohu.com/)作为评论系统。
*  也可以绑定自己的域名，如果没有域名，可以在[godaddy http://www.godaddy.com/](http://www.godaddy.com/)上将域名放入购物车等待降价，买之。
*  祝各位新年快乐！

---
