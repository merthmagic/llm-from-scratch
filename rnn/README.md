# RNN

循环神经网络

这里放一些循环神经网络的学习笔记和代码

## 阅读材料


https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neural-networks/

https://towardsdatascience.com/implementing-recurrent-neural-network-using-numpy-c359a0a68a67

https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-1/

https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-3/

## 可参考的代码库

https://github.com/gy910210/rnn-from-scratch

## LSMT
LSTM（长短期记忆网络）可以被理解为循环神经网络（RNN）的一种实现。实际上，LSTM是一种特殊类型的RNN，旨在解决传统RNN中存在的梯度消失和梯度爆炸等问题。LSTM网络通过引入门控机制，能够更好地捕捉时间序列数据中的长期依赖关系，因此在处理时间序列数据方面具有更好的性能。因此，可以将LSTM视为RNN的一种改进和扩展。

## Attention
RNN中可以使用attention机制来提高模型对输入序列的关注度，从而提高模型的性能。通过attention机制，模型可以动态地调整对输入序列中不同部分的关注程度，从而更好地捕捉输入序列中的信息。这种机制可以帮助提高RNN模型在处理长序列和复杂序列时的表现，并在许多自然语言处理和序列建模任务中取得了良好的效果。