English | [中文](README.zh-CN.md)

# LLM from scratch

> What I cannot create, I do not understand -- Richard Feynman

This repo records the principles and engineering techniques related to LLMs, and also includes content on DL (Deep Learning) and RL (Reinforcement Learning).

## Contents

In learning order:

1. [Numpy](numpy/) - arrays and matrix basics
2. [Word2Vec](word2vec/) - CBOW / Skip-Gram embeddings
3. [NPLM](nplm/) - Neural Probabilistic Language Model
4. [RNN](rnn/) - recurrent neural networks (from scratch / PyTorch)
5. [Seq2Seq](seq2seq/) - sequence-to-sequence models
6. [NER](ner/) - named entity recognition

## Directory layout

Every topic follows the same structure:

```text
<topic>/
├── README.md      # notes and docs
├── notebook/      # Jupyter notebooks
├── assets/        # images and other assets
├── data/          # datasets
└── *.py           # optional standalone scripts
```

## Setup

```bash
pip install -r requirements.txt
```

## References

- [GPT图解](https://book.douban.com/subject/36668702/)
- [Dive into Deep Learning](https://d2l.ai/)
- [深度学习进阶：自然语言处理](https://book.douban.com/subject/35225413/)
- Deep Learning for NLP
