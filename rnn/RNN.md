# RNN

这里记录rnn学习



## 阅读材料

https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neural-networks/



## 笔记

第一个需要ponder的问题，是否真的需要RNN这种结构? RNN能完成什么下游任务?

1. sentiment classification,  边长=>定长
2. image captioning, 定长=>变长
3.  translation， 变长=>变长

预测下一个词，用MLP实现会出现什么情况？

从MLP出发，推导出了RNN

从结构推导一下公式

$$
h(t) = f(h_{t-1},x_t)
$$

$h_{t-1}$ 是前一个状态, $x_t$ 是当前输入，$h_t$ 是当前的新状态

用 $\tanh$ 作为激活函数，循环神经元的权重是 $W_{hh}$, 输入神经元的的权重 $W_{xh}$  , 时刻 $t$ 的状态为:

$$
h_t = \tanh(W_{hh}h_{t-1}+W_{xh}x_t)
$$

用于计算输出的为

$$
y_t = W_{hy}h_t
$$


> 1. A single time step of the input is supplied to the network i.e. xt is supplied to the network
> 2. We then calculate its current state using a combination of the current input and the previous state i.e. we calculate ht
> 3. The current ht becomes ht-1 for the next time step
> 4. We can go as many time steps as the problem demands and combine the information from all the previous states
> 5. Once all the time steps are completed the final current state is used to calculate the output yt
> 6. The output is then compared to the actual output and the error is generated
> 7. The error is then backpropagated to the network to update the weights(we shall go into the details of backpropagation in further sections) and the network is trained



用hello这个单词训练，tokenizer是逐个字母切分，词表变成{h,e,l,o}