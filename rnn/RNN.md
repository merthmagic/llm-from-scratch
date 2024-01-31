# RNN

这里记录RNN学习



## 阅读材料

https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neural-networks/

https://github.com/CaptainE/RNN-LSTM-in-numpy

## 笔记

第一个需要ponder的问题，是否真的需要RNN这种结构? RNN能完成什么下游任务?

1. sentiment classification,  变长=>定长
2. image captioning, 定长=>变长
3.  translation， 变长=>变长

预测下一个词，用MLP实现会出现什么情况?

从MLP出发，推导出了RNN，原文和思考如下

>Let’s have a deeper network, where multiple hidden layers are present. So here, the input layer receives the input, the first hidden layer activations are applied and then these activations are sent to the next hidden layer, and successive activations through the layers to produce the output. Each hidden layer is characterized by its own weights and biases.

>Since each hidden layer has its own weights and activations, they behave independently. Now the objective is to identify the relationship between successive inputs. Can we supply the inputs to hidden layers? Yes we can!

>Here, the weights and bias of these hidden layers are different. And hence each of these layers behave independently and cannot be combined together. To combine these hidden layers together, we shall have the same weights and bias for these hidden layers.

>We can now combines these layers together, that the weights and bias of all the hidden layers is the same. All these hidden layers can be rolled in together in a single recurrent layer.

```text
如果只用MLP,处理一个序列，我们可以有多个hidden层，比如第一个hidden层去计算第一个输入，第二个hidden层去计算第二输入(这里原文中提到，input是可以传入hidden层的，第二个hidden层可以接受序列中的第二个token作为输入，同时也接收第一个hidden层的输出作为输入)，以此类推形成多个hidden层...
这里有个问题还没想明白，似乎这样可以达到RNN的效果，但GPT4回答说这样很难捕捉到序列中token的关系，文章原文说法是这样每个hidden层各自为战不能合并(每个hidden层都有自己的权重和偏置)

gtp4给了一个回答：

合并隐藏层并不是必需的，而是根据特定的应用场景和目标来决定的。下面是一些可能促使合并隐藏层的原因：

序列数据处理：如前所述，若任务涉及序列数据（例如时间序列数据、文本、语音等），使用循环神经网络（RNN）可以更好地捕捉数据中的时间依赖性。在RNN中，隐藏层的权重和偏置被共享，这样可以在序列的不同时间步中维持状态，并传递相关信息。

参数数量减少：在深度神经网络中，每增加一个隐藏层，就会显著增加网络的参数数量（权重和偏置）。如果网络中有许多层，参数数量会迅速增加，这可能导致过拟合以及计算资源的大量消耗。通过合并隐藏层，例如在RNN中使用相同的权重和偏置，可以显著减少模型的总参数数量，从而有助于减少过拟合风险并提高计算效率。

模型泛化能力：共享权重有助于提高模型的泛化能力。在处理序列数据时，模型学习到的是如何基于前一状态和当前输入来更新状态，而不是仅针对特定位置的输入。这使得模型能够更好地泛化到新的、未见过的序列数据。

内存效率：使用共享权重的循环层可以更高效地使用内存。与拥有大量独立参数的深层网络相比，循环层不需要为每一个时间步存储或更新一套新的参数，这使得模型更加内存高效。

学习长期依赖性：在序列处理任务中，RNN及其变体（如LSTM和GRU）被设计用来捕捉长距离的依赖关系，这是普通深层前馈网络难以做到的。

总之，是否合并隐藏层取决于要解决的问题类型以及所需的网络特性。合并隐藏层并使用循环结构通常是为了更好地处理序列数据，并使模型更加高效和具有泛化能力。然而，对于不涉及时间序列依赖性的任务，传统的多层前馈神经网络可能是更好的选择。

```

从结构推导一下公式

$$
h(t) = f(h_{t-1},\mathbf{x}_t)
$$

$h_{t-1}$ 是前一个状态, $\mathbf{x}_t$ 是当前输入, $h_t$ 是当前的新状态

用 $\tanh$ 作为激活函数，循环神经元的权重是 $W_{hh}$, 输入神经元的的权重 $W_{xh}$  , 时刻 $t$ 的状态为:

$$
h_t = \tanh(W_{hh}h_{t-1}+W_{xh}\mathbf{x}_t)
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