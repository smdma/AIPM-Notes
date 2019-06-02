> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 https://blog.csdn.net/lu_embedded/article/details/81100704 版权声明：开心源自分享，快乐源于生活 —— 分享技术，传递快乐。转载文章请注明出处，谢谢！ https://blog.csdn.net/luckydarcy/article/details/81100704

　　本文从 “**是什么**”、“**为什么**”、“**怎么办**”、“**好不好**” 四个维度来介绍 GitBook，带你从黑暗之中走出来，get 这种美妙的写作方式。

------

## <a></a>是什么？

　　在我认识 GitBook 之前，我已经在使用 Git 了，毋容置疑，Git 是目前世界上最先进的分布式版本控制系统。

　　我认为 Git 不仅是程序员管理代码的工具，它的分布式协作方式同样适用于很多场合，其中一个就是写作（这会是一个引起社会变革的伟大的工具！）。所以在我发现 GitBook 之前，实际上我已经无数次想象过它的使用场景了。

　　咋一看 GitBook 的名字，你可能会认为它是关于 Git 的一本书。而当你有所了解之后，你也许会认为它是一个使用 Git 构建电子书的工具。其实不然，GitBook 与 Git 的关系，就像雷锋塔和雷锋那样，没有一点关系！

<center>![](https://img-blog.csdn.net/20180718161255281?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1Y2t5ZGFyY3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)</center>

　　实际上，GitBook 是一个基于 Node.js 的命令行工具，支持 Markdown 和 AsciiDoc 两种语法格式，可以输出 HTML、PDF、eBook 等格式的电子书。所以我更喜欢把 GitBook 定义为文档格式转换工具。

　　所以，GitBook 不是 Markdown 编辑工具，也不是 Git 版本管理工具。市面上我们可以找到很多 Markdown 编辑器，比如 Typora、MacDown、Bear、MarkdownPad、MarkdownX、JetBrains’s IDE（需要安装插件）、Atom、简书、CSDN 以及 GitBook 自家的 GitBook Editor 等等。

<center>![](https://img-blog.csdn.net/20180718161741325?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1Y2t5ZGFyY3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)</center>

　　**但 GitBook 又与 Markdown 和 Git 息息相关**，因为只有将它们结合起来使用，才能将它们的威力发挥到极致！因此，通常我们会选择合适的 Markdown 编辑工具以获得飞一般的写作体验；使用 GitBook 管理文档，预览、制作电子书；同时通过 Git 管理书籍内容的变更，并将其托管到云端（比如 GitHub、GitLab、码云，或者是自己搭建的 Git 服务器），实现多人协作。

　　实际上，GitBook Editor 对于新手来说是个不错的选择，它集成了 GitBook、Git、Markdown 等功能，还支持将书籍同步到 gitbook.com 网站，使我们可以很方便地编辑和管理书籍。但是不幸的是，GitBook Editor 的注册和登录需要翻墙，即便注册成功了也可能登录不上，似乎是因为网站最近在升级。

<center>![](https://img-blog.csdn.net/20180718171731114?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1Y2t5ZGFyY3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)</center>

　　因此，我推荐，也是我目前使用的搭配是 `GitBook + Typora + Git`。

------

## <a></a>为什么？

　　通常，我们最开始学习和使用的办公软件就是 Word、Excel 和 PowerPoint。这里不是说它们已经过时了，不是说 GitBook 能够替代它们。

　　相反，Microsoft 的办公软件很优秀并且经受了时间的考验，但是正因为它功能丰富，导致稍显臃肿（二八定律：80% 的时间里我们只会只用 20% 的功能），同时因为它存在以二进制格式保存、软件不兼容、格式不兼容、难以进行版本控制、难以实时分享预览、难以多人协作等短板。而这恰恰是 GitBook + Markdown + Git 的长处。

　　简单来说，GitBook + Markdown + Git 带来的好处有：

> - 语法简单
> - 兼容性强
> - 导出方便
> - 专注内容
> - 团队协作

　　所以，如果你和我一样，不满足于传统的写作方式，正在寻找一种令人愉悦的写作方式，那么该尝试使用 GitBook 啦！

<center>![](https://img-blog.csdn.net/20180718174333570?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1Y2t5ZGFyY3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0)</center>

　　当然，GitBook 不是万能的，当我们需要复杂排版时，依然需要依托于 Word 等工具。但不用担心，因为我们可以把 Markdown 格式的文档导出为 Word 格式，再进一步加工。

------

## <a></a>怎么办？

### <a></a>怎么安装

　　当你听了我的怂恿，并决定尝试使用 GitBook 的时候，首先面临的问题是 —— 如何搭建 GitBook 环境？

　　因为 GitBook 是基于 Node.js，所以我们首先需要安装 Node.js（下载地址：[https://nodejs.org/en/download/](https://nodejs.org/en/download/)），找到对应平台的版本安装即可。

　　现在安装 Node.js 都会默认安装 npm（node 包管理工具），所以我们不用单独安装 npm，打开命令行，执行以下命令安装 GitBook：

```
npm install -g gitbook-cli
```

　　安装完之后，就会多了一个 **gitbook** 命令（如果没有，请确认上面的命令是否加了 `-g`）。

　　上面我推荐的是 GitBook + Typora + Git，所以你还需要安装 Typora（一个很棒的支持 macOS、Windows、Linux 的 Markdown 编辑工具）和 Git 版本管理工具。戳下面：

> - Typora 下载地址：[https://typora.io/](https://typora.io/)
> - Git 下载地址：[https://git-scm.com/downloads](https://git-scm.com/downloads)

　　Typora 的安装很简单，难点在于需要翻墙才能下载（当然你也可以找我要）。Git 的安装也很简单，但要用好它需要不少时间，这里就不展开了（再讲下去你就要跑啦）。

### <a></a>怎么使用

　　想象一下，现在你准备构建一本书籍，你在硬盘上新建了一个叫 mybook 的文件夹，按照以前的做法，你会新建一个 Word 文档，写上标题，然后开始巴滋巴滋地笔耕。但是现在有了 GitBook，你首先要做的是在 mybook 文件夹下执行以下命令：

```
$ gitbook init
```

　　执行完后，你会看到多了两个文件 —— README.md 和 SUMMARY.md，它们的作用如下：

> - README.md —— 书籍的介绍写在这个文件里
> - SUMMARY.md —— 书籍的目录结构在这里配置

　　这时候，我们启动恭候多时的 Typora 来编辑这两个文件了：

<center>![](https://img-blog.csdn.net/2018071818281621?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1Y2t5ZGFyY3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)</center>

　　编辑 SUMMARY.md 文件，内容修改为：

```
# 目录

* [前言](README.md)
* [第一章](Chapter1/README.md)
  * [第1节：衣](Chapter1/衣.md)
  * [第2节：食](Chapter1/食.md)
  * [第3节：住](Chapter1/住.md)
  * [第4节：行](Chapter1/行.md)
* [第二章](Chapter2/README.md)
* [第三章](Chapter3/README.md)
* [第四章](Chapter4/README.md)

```

　　然后我们回到命令行，在 mybook 文件夹中再次执行 `gitbook init` 命令。GitBook 会查找 SUMMARY.md 文件中描述的目录和文件，如果没有则会将其创建。

　　Typora 是所见即所得（实时渲染）的 Markdown 编辑器，这时候它是这样的：

<center>![](https://img-blog.csdn.net/20180718185415241?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1Y2t5ZGFyY3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)</center>

　　接着我们执行 `gitbook serve` 来预览这本书籍，执行命令后会对 Markdown 格式的文档进行转换，默认转换为 html 格式，最后提示 “Serving book on [http://localhost:4000](http://localhost:4000)”。嗯，打开浏览器看一下吧：

<center>![](https://img-blog.csdn.net/20180718185251753?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1Y2t5ZGFyY3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)</center>

　　当你写得差不多，你可以执行 `gitbook build` 命令构建书籍，默认将生成的静态网站输出到 _book 目录。实际上，这一步也包含在 `gitbook serve` 里面，因为它们是 HTML，所以 GitBook 通过 Node.js 给你提供服务了。
　　当然，build 命令可以指定路径：

```
$ gitbook build [书籍路径] [输出路径]
```

　　serve 命令也可以指定端口：

```
$ gitbook serve --port 2333
```

　　你还可以生成 PDF 格式的电子书：

```
$ gitbook pdf ./ ./mybook.pdf
```

　　生成 epub 格式的电子书：

```
$ gitbook epub ./ ./mybook.epub
```

　　生成 mobi 格式的电子书：

```
$ gitbook mobi ./ ./mybook.mobi
```

　　如果生成不了，你可能还需要安装一些工具，比如 ebook-convert。或者在 Typora 中安装 Pandoc 进行导出。

　　除此之外，别忘了还可以用 Git 做版本管理呀！在 mybook 目录下执行 `git init` 初始化仓库，执行 `git remote add` 添加远程仓库（你得先在远端建好）。接着就可以愉快地 commit，push，pull … 啦！

------

## <a></a>好不好？

　　额…… 你觉得好不好嘛？

　　反正我觉得挺好的，特别是对我这种懒得排版，又想随时随地写作的宝宝来说。而且能够查看每个版本内容变更的情况，同时又避免了硬盘单一故障带来的风险。

------

（全剧终）

<link href="https://csdnimg.cn/release/phoenix/mdeditor/markdown_views-7f770a53f2.css" rel="stylesheet">