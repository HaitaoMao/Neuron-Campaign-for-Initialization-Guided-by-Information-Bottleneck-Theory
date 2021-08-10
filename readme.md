# Neuron Campaign for Initialization Guided by Information Bottleneck Theory

This repository is the official implementation of Neuron Campaign for Initialization Guided by Information Bottleneck Theory. 

## Abstract

Initialization plays a critical role in the training of deep neural net-works (DNN). Existing initialization strategies mainly focus on stabilizing the training process to mitigate gradient vanish/explosion problem. However, these initialization methods are lacking in consideration about how to enhance generalization ability. The In-formation Bottleneck (IB) theory is a well-known understanding framework to provide an explanation about the generalization of DNN. Guided by the insights provided by IB theory, we design two criteria for better initializing DNN. And we further design a neuron campaign initialization algorithm to efficiently select a good initialization for a neural network on a given dataset. The experiments on MNIST dataset show that our method can lead to a better generalization performance with faster convergence.

## Training

To train the models (MLP-2 by default) in the paper, run this command:

```
python main.py
```
## Model
This table shows the network architecture of DNN model.
| Model | Hidden layer dimension |
| ----- | ---------------------- |
| MLP-2 | [100]                  |
| MLP-3 | [256, 100]             |
| MLP-5 | [32, 32, 32, 32]       |



## License
All content in this repository is licensed under the [MIT license](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).