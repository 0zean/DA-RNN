# DA-RNN
 Dual-Stage Attention-Based Recurrent Neural Network for financial time-series prediction


PyTorch implementation of the [Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971)
using [Kaelzhang's library](https://github.com/kaelzhang/DA-RNN-in-Tensorflow-2-and-PyTorch) implementation of the network 


Extensive feature set generation is done by combining correlated assets, technical analysis of the open price, and windowed rough path signatures. Only the open price and lagged close price are used in the calculation of technical indicators as knowledge of the close price is not known in real time.


In sample results:

<img src="https://github.com/0zean/DA-RNN/blob/main/Figure_1.png?raw=true" width=75% height=75%>

Out of sample results:

<img src="https://github.com/0zean/DA-RNN/blob/main/Figure_2.png?raw=true" width=75% height=75%>
