# Numpy基础

这里记录[Numpy](https://numpy.org/)(Numerical Python)的基础操作


## Quick start

参考[NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
摘录了文档并加入一些自己的笔记
### 安装
在conda下
```bash
conda install numpy
```
在pip下
```bash
pip install numpy
```
jupyter
```bash
!pip install numpy
```

### 导入
```python
import numpy as np
```

### Python list和 Numpy array区别
在python中，list中的数据可以是不同类型；在Numpy array中，必须是同质(homogeneous).如果数组不是同质的，那么对数组进行的数学操作将会非常低效。

>NumPy arrays are faster and more compact than Python lists. An array consumes less memory and is convenient to use. NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. This allows the code to be optimized even further.

NumPy数组比Python列表更快，更紧凑。数组占用更少的内存并且使用起来更方便。NumPy使用更少的内存来存储数据，并提供了一种指定数据类型的机制。这使得代码可以进一步优化。

### 什么是array
>An array is a central data structure of the NumPy library. An array is a grid of values and it contains information about the raw data, how to locate an element, and how to interpret an element. It has a grid of elements that can be indexed in various ways. The elements are all of the same type, referred to as the array dtype.

数组是NumPy库的核心数据结构。数组是一个数值网格，它包含有关原始数据的信息，如何定位一个元素以及如何解释一个元素。它包含一系列可以以多种方式进行索引的元素。这些元素都是相同类型的，被称为数组的数据类型（dtype）。

>An array can be indexed by a tuple of nonnegative integers, by booleans, by another array, or by integers. The rank of the array is the number of dimensions. The shape of the array is a tuple of integers giving the size of the array along each dimension.

数组可以通过<mark>非负整数的元组、布尔值、另一个数组或整数进行索引</mark>。数组的秩（rank）是指数组的维度数量。数组的形状（shape）是一个由<mark>整数组成的元组</mark>，它给出了数组沿着每个维度的大小。