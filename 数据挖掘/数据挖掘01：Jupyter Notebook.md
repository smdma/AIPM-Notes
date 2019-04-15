## 速查
[Jupyter Notebook官方文档](https://jupyter-notebook.readthedocs.io/en/stable/)

## 1、库的安装

## 学习目标

- 目标
  - 搭建好数据挖掘基础阶段的环境
- 应用
  - 无

整个数据挖掘基础阶段会用到Matplotlib、Numpy、Pandas、Ta-Lib等库，为了统一版本号在环境中使用，将所有的库及其版本放到了文件requirements.txt当中，然后统一安装

**新建一个用于人工智能环境的虚拟环境**

```
mkvirtualenv -p /usr/local/bin/python3 ai
matplotlib==2.2.2
numpy==1.14.2
pandas==0.20.3
TA-Lib==0.4.16
tables==3.4.2
jupyter==1.0.0
```

使用pip命令安装

```
pip install -r requirements.txt
```

如果Ta-Lib安装出现问题，需要先安装依赖库，按照以下步骤安装：

```python
# 获取源码库
sudo wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
# 解压进入目录
tar -zxvf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
# 编译安装
sudo ./configure --prefix=/usr  
sudo make
sudo make install
# 重新安装python的TA-Lib库
pip install TA-Lib
```



# 2 Jupyter Notebook使用

## 学习目标

- 目标
  - 学会使用Jupyter Notebook编写运行代码
- 应用
  - 创建文件
  - 操作cell
  - 运行操作
- 内容预览
  - 1.2.1 Jupyter Notebook介绍
  - 1.2.2 为什么使用Jupyter Notebook?
  - 1.2.3 Jupyter Notebook的使用-helloworld
    - 1 界面启动、创建文件
    - 2 cell操作
    - 3 markdown演示

## 1.2.1 Jupyter Notebook介绍

Jupyter项目是一个非盈利的开源项目，源于2014年的ipython项目，并逐渐发展为支持跨所有编程语言的交互式数据科学计算的工具。

- Jupyter Notebook，原名IPython Notbook，是IPython的加强网页版，一个开源Web应用程序
- 名字源自Julia、Python 和 R（数据科学的三种开源语言）
- 是一款程序员和科学工作者的**编程/文档/笔记/展示**软件
- **.ipynb**文件格式是用于计算型叙述的**JSON文档格式**的正式规范

![jupyternotebook](../images/jupyternotebook.png)

## 1.2.2 为什么使用Jupyter Notebook?

- 传统软件开发：工程／目标明确
  - 需求分析，设计架构，开发模块，测试
- 数据挖掘：艺术／目标不明确
  - 目的是具体的洞察目标，而不是机械的完成任务
  - 通过执行代码来理解问题
  - 迭代式地改进代码来改进解决方法

实时运行的代码、叙事性的文本和可视化被整合在一起，方便使用代码和数据来讲述故事

**对比Jupyter Notebook和Pycharm**

- 画图

![img](./images/展示1.png)

- 数据展示

  ![img](./images/展示2.png)

  - 总结：Jupyter Notebook 相比 Pycharm、Ipython在画图和数据展示方面更有优势。

## 1.2.3 Jupyter Notebook的使用-helloworld

### 1 界面启动、创建文件

- 界面启动

环境搭建好后，本机输入jupyter notebook命令，会自动弹出浏览器窗口打开Jupyter Notebook

```python
# 进入虚拟环境
workon ai
# 输入命令
jupyter notebook
```

本地notebook的默认URL为：<http://localhost:8888>

想让notebook打开指定目录，只要进入此目录后执行命令即可

![notebook1](../images/notebook1.png)

- 新建notebook文档
  - notebook的文档格式是`.ipynb`

![img](../images/createnotebook.png)

- 内容界面操作-helloworld

标题栏：点击标题（如Untitled）修改文档名 菜单栏

- 导航-File-Download as，另存为其他格式
- 导航-Kernel
  - Interrupt，中断代码执行（程序卡死时）
  - Restart，重启Python内核（执行太慢时重置全部资源）
  - Restart & Clear Output，重启并清除所有输出
  - Restart & Run All，重启并重新运行所有代码

![controlnotebook](../images/jupyter_helloworld.png)

### 2 cell操作

什么是cell？

**cell**：一对In Out会话被视作一个代码单元，称为cell

Jupyter支持两种模式：

- 编辑模式（Enter）
  - 命令模式下`回车Enter`或`鼠标双击`cell进入编辑模式
  - 可以**操作cell内文本**或代码，剪切／复制／粘贴移动等操作
- 命令模式（Esc）
  - 按`Esc`退出编辑，进入命令模式
  - 可以**操作cell单元本身**进行剪切／复制／粘贴／移动等操作

#### 1）鼠标操作

![工具栏cell](../images/工具栏cell.png)

#### 2）快捷键操作

- 两种模式通用快捷键
  - **Shift+Enter，执行本单元代码，并跳转到下一单元**
  - **Ctrl+Enter，执行本单元代码，留在本单元**

cell行号前的 * ，表示代码正在运行

- 命令模式

  ：按ESC进入

  - `Y`，cell切换到Code模式
  - `M`，cell切换到Markdown模式
  - `A`，在当前cell的上面添加cell
  - `B`，在当前cell的下面添加cell
  - `双击D`：删除当前cell
  - `Z`，回退
  - `L`，为当前cell加上行号 <!--
  - `Ctrl+Shift+P`，对话框输入命令直接运行
  - 快速跳转到首个cell，`Crtl+Home`
  - 快速跳转到最后一个cell，`Crtl+End` -->

- 编辑模式

  ：按Enter进入

  - 多光标操作：`Ctrl键点击鼠标`（Mac:CMD+点击鼠标）
  - 回退：`Ctrl+Z`（Mac:CMD+Z）
  - 重做：`Ctrl+Y`（Mac:CMD+Y)
  - 补全代码：变量、方法后跟`Tab键`
  - 为一行或多行代码添加/取消注释：`Ctrl+/`（Mac:CMD+/）
  - 屏蔽自动输出信息：可在最后一条语句之后加一个分号

