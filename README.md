English | [中文](README.zh-CN.md)

# LLM from scratch

> What I cannot create, I do not understand -- Richard Feynman

This repo records my study notes on LLM fundamentals, plus some deep learning basics.

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

- GPT图解
- Dive into Deep Learning
- 深度学习进阶：自然语言处理
- Deep Learning for NLP
