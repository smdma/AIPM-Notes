# Seaborn

Seaborn是一个以matplotlib为底层，更容易定制化作图的库。
安装seaborn的方法：
pip install seaborn

**Pandas与Seaborn之间的区别：**
1. 在只需要简单地作图时直接用Pandas，但要想做出更加吸引人，更丰富的图就可以使用Seaborn
2. Pandas的作图函数并没有太多的参数来调整图形，所以你必须要深入了解matplotlib
3. Seaborn的作图函数中提供了大量的参数来调整图形，所以并不需要太深入了解matplotlib
4. Seaborn的API：<http://seaborn.pydata.org/api.html>

## 0、速查
[Seaborn官方文档](http://seaborn.pydata.org/api.html)

### 关系图

| [`relplot`](http://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot)（[x，y，hue，size，style，data，row，...]） | 用于在FacetGrid上绘制关系图的图级界面。 |
| ------------------------------------------------------------ | --------------------------------------- |
| [`scatterplot`](http://seaborn.pydata.org/generated/seaborn.scatterplot.html#seaborn.scatterplot)（[x，y，hue，style，size，data，...]） | 绘制具有多个语义分组可能性的散点图。    |
| [`lineplot`](http://seaborn.pydata.org/generated/seaborn.lineplot.html#seaborn.lineplot)（[x，y，hue，size，style，data，...]） | 绘制一个线图，可能有几个语义分组。      |



### 分类图

| [`catplot`](http://seaborn.pydata.org/generated/seaborn.catplot.html#seaborn.catplot)（[x，y，hue，data，row，col，...]） | 用于将分类图绘制到FacetGrid上的图级界面。 |
| ------------------------------------------------------------ | ----------------------------------------- |
| [`stripplot`](http://seaborn.pydata.org/generated/seaborn.stripplot.html#seaborn.stripplot)（[x，y，hue，data，order，...]） | 绘制一个散点图，其中一个变量是分类的。    |
| [`swarmplot`](http://seaborn.pydata.org/generated/seaborn.swarmplot.html#seaborn.swarmplot)（[x，y，hue，data，order，...]） | 绘制具有非重叠点的分类散点图。            |
| [`boxplot`](http://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot)（[x，y，hue，data，order，hue_order，...]） | 绘制方框图以显示与类别相关的分布。        |
| [`violinplot`](http://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot)（[x，y，hue，data，order，...]） | 绘制箱线图和核密度估计的组合。            |
| [`boxenplot`](http://seaborn.pydata.org/generated/seaborn.boxenplot.html#seaborn.boxenplot)（[x，y，hue，data，order，...]） | 为较大的数据集绘制增强的框图。            |
| [`pointplot`](http://seaborn.pydata.org/generated/seaborn.pointplot.html#seaborn.pointplot)（[x，y，hue，data，order，...]） | 使用散点图字形显示点估计值和置信区间。    |
| [`barplot`](http://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot)（[x，y，hue，data，order，hue_order，...]） | 将点估计和置信区间显示为矩形条。          |
| [`countplot`](http://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot)（[x，y，hue，data，order，...]） | 使用条形显示每个分类箱中的观察计数。      |



### 分布图

| [`jointplot`](http://seaborn.pydata.org/generated/seaborn.jointplot.html#seaborn.jointplot)（x，y [，data，kind，stat_func，...]） | 用双变量和单变量图绘制两个变量的图。   |
| ------------------------------------------------------------ | -------------------------------------- |
| [`pairplot`](http://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot)（data [，hue，hue_order，palette，...]） | 绘制数据集中的成对关系。               |
| [`distplot`](http://seaborn.pydata.org/generated/seaborn.distplot.html#seaborn.distplot)（a [，bins，hist，kde，rug，fit，...]） | 灵活绘制单变量观测分布。               |
| [`kdeplot`](http://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot)（数据[，data2，shade，vertical，...]） | 拟合并绘制单变量或双变量核密度估计值。 |
| [`rugplot`](http://seaborn.pydata.org/generated/seaborn.rugplot.html#seaborn.rugplot)（a [，身高，轴，斧头]） | 将数组中的数据点绘制为轴上的棒。       |



### 回归图

| [`lmplot`](http://seaborn.pydata.org/generated/seaborn.lmplot.html#seaborn.lmplot)（x，y，data [，hue，col，row，palette，...]） | 绘图数据和回归模型适用于FacetGrid。 |
| ------------------------------------------------------------ | ----------------------------------- |
| [`regplot`](http://seaborn.pydata.org/generated/seaborn.regplot.html#seaborn.regplot)（x，y [，data，x_estimator，x_bins，...]） | 绘图数据和线性回归模型拟合。        |
| [`residplot`](http://seaborn.pydata.org/generated/seaborn.residplot.html#seaborn.residplot)（x，y [，data，lowess，x_partial，...]） | 绘制线性回归的残差。                |



### 矩阵图

| [`heatmap`](http://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap)（数据[，vmin，vmax，cmap，center，...]） | 将矩形数据绘制为颜色编码矩阵。   |
| ------------------------------------------------------------ | -------------------------------- |
| [`clustermap`](http://seaborn.pydata.org/generated/seaborn.clustermap.html#seaborn.clustermap)（data [，pivot_kws，method，...]） | 将矩阵数据集绘制为分层聚类热图。 |



### 多绘图网格

#### 小平面网格

| [`FacetGrid`](http://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid)（data [，row，col，hue，col_wrap，...]） | 用于绘制条件关系的多图网格。                             |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| [`FacetGrid.map`](http://seaborn.pydata.org/generated/seaborn.FacetGrid.map.html#seaborn.FacetGrid.map)（func，* args，** kwargs） | 将绘图功能应用于每个方面的数据子集。                     |
| [`FacetGrid.map_dataframe`](http://seaborn.pydata.org/generated/seaborn.FacetGrid.map_dataframe.html#seaborn.FacetGrid.map_dataframe)（func，* args，** kwargs） | 喜欢`.map`但是将args作为字符串传递并在kwargs中插入数据。 |

#### 配对网格

| [`PairGrid`](http://seaborn.pydata.org/generated/seaborn.PairGrid.html#seaborn.PairGrid)（data [，hue，hue_order，palette，...]） | 用于绘制数据集中成对关系的子图网格。       |
| ------------------------------------------------------------ | ------------------------------------------ |
| [`PairGrid.map`](http://seaborn.pydata.org/generated/seaborn.PairGrid.map.html#seaborn.PairGrid.map)（func，** kwargs） | 在每个子图中绘制具有相同功能的图。         |
| [`PairGrid.map_diag`](http://seaborn.pydata.org/generated/seaborn.PairGrid.map_diag.html#seaborn.PairGrid.map_diag)（func，** kwargs） | 在每个对角线子图上绘制具有单变量函数的图。 |
| [`PairGrid.map_offdiag`](http://seaborn.pydata.org/generated/seaborn.PairGrid.map_offdiag.html#seaborn.PairGrid.map_offdiag)（func，** kwargs） | 在非对角线子图上绘制具有双变量函数的图。   |
| [`PairGrid.map_lower`](http://seaborn.pydata.org/generated/seaborn.PairGrid.map_lower.html#seaborn.PairGrid.map_lower)（func，** kwargs） | 在下对角线子图上绘制具有双变量函数的图。   |
| [`PairGrid.map_upper`](http://seaborn.pydata.org/generated/seaborn.PairGrid.map_upper.html#seaborn.PairGrid.map_upper)（func，** kwargs） | 在上对角线子图上绘制具有双变量函数的图。   |

#### 联合网格

| [`JointGrid`](http://seaborn.pydata.org/generated/seaborn.JointGrid.html#seaborn.JointGrid)（x，y [，data，height，ratio，...]） | 用于绘制具有边际单变量图的双变量图的网格。 |
| ------------------------------------------------------------ | ------------------------------------------ |
| [`JointGrid.plot`](http://seaborn.pydata.org/generated/seaborn.JointGrid.plot.html#seaborn.JointGrid.plot)（joint_func，marginal_func [，...]） | 画出完整情节的捷径。                       |
| [`JointGrid.plot_joint`](http://seaborn.pydata.org/generated/seaborn.JointGrid.plot_joint.html#seaborn.JointGrid.plot_joint)（func，** kwargs） | 绘制x和y的双变量图。                       |
| [`JointGrid.plot_marginals`](http://seaborn.pydata.org/generated/seaborn.JointGrid.plot_marginals.html#seaborn.JointGrid.plot_marginals)（func，** kwargs） | 分别绘制x和y的单变量图。                   |



### 样式控制

| [`set`](http://seaborn.pydata.org/generated/seaborn.set.html#seaborn.set)（[context，style，palette，font，...]） | 一步设定美学参数。                           |
| ------------------------------------------------------------ | -------------------------------------------- |
| [`axes_style`](http://seaborn.pydata.org/generated/seaborn.axes_style.html#seaborn.axes_style)（[style，rc]） | 返回参数字典，用于绘图的美学风格。           |
| [`set_style`](http://seaborn.pydata.org/generated/seaborn.set_style.html#seaborn.set_style)（[style，rc]） | 设定地块的审美风格。                         |
| [`plotting_context`](http://seaborn.pydata.org/generated/seaborn.plotting_context.html#seaborn.plotting_context)（[context，font_scale，rc]） | 返回参数dict以缩放图形的元素。               |
| [`set_context`](http://seaborn.pydata.org/generated/seaborn.set_context.html#seaborn.set_context)（[context，font_scale，rc]） | 设置绘图上下文参数。                         |
| [`set_color_codes`](http://seaborn.pydata.org/generated/seaborn.set_color_codes.html#seaborn.set_color_codes)（[调色板]） | 更改matplotlib颜色缩写词的解释方式。         |
| [`reset_defaults`](http://seaborn.pydata.org/generated/seaborn.reset_defaults.html#seaborn.reset_defaults)（） | 将所有RC参数恢复为默认设置。                 |
| [`reset_orig`](http://seaborn.pydata.org/generated/seaborn.reset_orig.html#seaborn.reset_orig)（） | 将所有RC参数恢复为原始设置（尊重自定义rc）。 |



### 调色板

| [`set_palette`](http://seaborn.pydata.org/generated/seaborn.set_palette.html#seaborn.set_palette)（palette [，n_colors，desat，...]） | 使用seaborn调色板设置matplotlib颜色循环。   |
| ------------------------------------------------------------ | ------------------------------------------- |
| [`color_palette`](http://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette)（[palette，n_colors，desat]） | 返回定义调色板的颜色列表。                  |
| [`husl_palette`](http://seaborn.pydata.org/generated/seaborn.husl_palette.html#seaborn.husl_palette)（[n_colors，h，s，l]） | 在HUSL色调空间中获得一组均匀间隔的颜色。    |
| [`hls_palette`](http://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette)（[n_colors，h，l，s]） | 在HLS色调空间中获取一组均匀间隔的颜色。     |
| [`cubehelix_palette`](http://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html#seaborn.cubehelix_palette)（[n_colors，start，rot，...]） | 从cubehelix系统制作顺序调色板。             |
| [`dark_palette`](http://seaborn.pydata.org/generated/seaborn.dark_palette.html#seaborn.dark_palette)（颜色[，n_colors，反向，......]） | 制作一个从黑暗到混合的顺序调色板`color`。   |
| [`light_palette`](http://seaborn.pydata.org/generated/seaborn.light_palette.html#seaborn.light_palette)（颜色[，n_colors，反向，......]） | 制作一个从光到混合的顺序调色板`color`。     |
| [`diverging_palette`](http://seaborn.pydata.org/generated/seaborn.diverging_palette.html#seaborn.diverging_palette)（h_neg，h_pos [，s，l，sep，...]） | 在两种HUSL颜色之间制作不同的调色板。        |
| [`blend_palette`](http://seaborn.pydata.org/generated/seaborn.blend_palette.html#seaborn.blend_palette)（colors [，n_colors，as_cmap，input]） | 制作一个混合颜色列表的调色板。              |
| [`xkcd_palette`](http://seaborn.pydata.org/generated/seaborn.xkcd_palette.html#seaborn.xkcd_palette)（颜色） | 使用xkcd颜色调查中的颜色名称制作调色板。    |
| [`crayon_palette`](http://seaborn.pydata.org/generated/seaborn.crayon_palette.html#seaborn.crayon_palette)（颜色） | 用Crayola蜡笔制作一个带有颜色名称的调色板。 |
| [`mpl_palette`](http://seaborn.pydata.org/generated/seaborn.mpl_palette.html#seaborn.mpl_palette)（name [，n_colors]） | 从matplotlib调色板返回离散颜色。            |

### 调色板小部件

| [`choose_colorbrewer_palette`](http://seaborn.pydata.org/generated/seaborn.choose_colorbrewer_palette.html#seaborn.choose_colorbrewer_palette)（data_type [，as_cmap]） | 从ColorBrewer集中选择一个调色板。           |
| ------------------------------------------------------------ | ------------------------------------------- |
| [`choose_cubehelix_palette`](http://seaborn.pydata.org/generated/seaborn.choose_cubehelix_palette.html#seaborn.choose_cubehelix_palette)（[as_cmap]） | 启动交互式小部件以创建顺序cubehelix调色板。 |
| [`choose_light_palette`](http://seaborn.pydata.org/generated/seaborn.choose_light_palette.html#seaborn.choose_light_palette)（[input，as_cmap]） | 启动交互式小部件以创建轻型顺序调色板。      |
| [`choose_dark_palette`](http://seaborn.pydata.org/generated/seaborn.choose_dark_palette.html#seaborn.choose_dark_palette)（[input，as_cmap]） | 启动交互式小部件以创建暗序连接调色板。      |
| [`choose_diverging_palette`](http://seaborn.pydata.org/generated/seaborn.choose_diverging_palette.html#seaborn.choose_diverging_palette)（[as_cmap]） | 启动交互式小部件以选择不同的调色板。        |

### 实用功能

| [`load_dataset`](http://seaborn.pydata.org/generated/seaborn.load_dataset.html#seaborn.load_dataset)（name [，cache，data_home]） | 从在线存储库加载数据集（需要互联网）。 |
| ------------------------------------------------------------ | -------------------------------------- |
| [`despine`](http://seaborn.pydata.org/generated/seaborn.despine.html#seaborn.despine)（[fig，ax，top，right，left，bottom，...]） | 从图中移除顶部和右侧脊柱。             |
| [`desaturate`](http://seaborn.pydata.org/generated/seaborn.desaturate.html#seaborn.desaturate)（颜色，道具） | 将颜色的饱和度通道减少百分之几。       |
| [`saturate`](http://seaborn.pydata.org/generated/seaborn.saturate.html#seaborn.saturate)（颜色） | 返回具有相同色调的完全饱和的颜色。     |
| [`set_hls_values`](http://seaborn.pydata.org/generated/seaborn.set_hls_values.html#seaborn.set_hls_values)（颜色[，h，l，s]） | 独立操作颜色的h，l或s通道。            |



## 直方图的绘制
```python
import matplotlib.pyplot as plt

import seaborn as sns #一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式

%matplotlib inline # 为了在jupyter notebook里作图，需要用到这个命令
sns.set_style('dark') # 该图使用黑色为背景色
sns.distplot(births['prglngth'], kde=False) # kde默认为True，会展示一条密度曲线，为False时不会展示该曲线。
sns.axlabel('Birth number', 'Frequency') # 设置X轴和Y轴的坐标含义

sns.plt.show()
```

## 箱型图的绘制
```python
# 以birthord作为x轴，agepreg作为y轴，做一个箱型图
sns.boxplot(x='birthord', y='agepreg', data=births)

sns.plt.show()
```

## 多变量作图
- seaborn可以一次性两两组合多个变量做出多个对比图，有n个变量，就会做出一个n × n个格子的图，譬如有2个变量，就会产生4个格子，每个格子就是两个变量之间的对比图。
- 相同的两个变量之间（var1  vs  var1 和 var2  vs  var2）以直方图展示，不同的变量则以散点图展示（var1  vs  var2 和var2  vs  var1）
- 要注意的是数据中不能有NaN（缺失的数据），否则会报错
```python
sns.pairplot(births, vars=['agepreg', 'prglngth','birthord'])

sns.plt.show()
```
