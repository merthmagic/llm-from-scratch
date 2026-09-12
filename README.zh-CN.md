[English](README.md) | 中文

# LLM from scratch

> What I cannot create, I do not understand -- Richard Feynman

这个 repo 主要记录 LLM 相关的基础知识学习，也补充一些 Deep Learning 相关内容.

## Contents

按学习顺序排列：

1. [Numpy](numpy/) - 数组与矩阵基础
2. [Word2Vec](word2vec/) - CBOW / Skip-Gram 词向量
3. [NPLM](nplm/) - 神经概率语言模型
4. [RNN](rnn/) - 循环神经网络（从零实现 / PyTorch）
5. [Seq2Seq](seq2seq/) - 序列到序列模型
6. [NER](ner/) - 命名实体识别

## 目录约定

每个主题目录结构保持一致：

```text
<topic>/
├── README.md      # 笔记与说明
├── notebook/      # Jupyter notebook
├── assets/        # 图片等资源
├── data/          # 数据集
└── *.py           # 可选的独立脚本
```

## 环境准备

```bash
pip install -r requirements.txt
```

## Reference

- GPT图解
- Dive into Deep Learning
- 深度学习进阶：自然语言处理
- Deep Learning for NLP
