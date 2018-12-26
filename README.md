# Medium - Math & Neural Network

This code is part of my post on **[medium](https://medium.com/@omaraflak/math-neural-network-from-scratch-in-python-d6da9f29ce65)**.

# Run it

```shell
python example_xor.py
python example_conv.py
```



----

> Make your own machine learning library.



![img](./Pic/1.jpg)

Photo by [Mathew Schwartz](https://unsplash.com/@cadop?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral)

In this post we will go through the math of machine learning and code from scratch, in Python, a small library to build neural networks with a variety of layers (Fully Connected, Convolutional, etc.). Eventually, we will be able to write something like :

> 在这篇文章中，我们将在Python中从头开始了解用于构建具有各种层神经网络（完全连接，卷积等）的小型库中的机器学习和代码。最终，我们将能够写出如下内容：
>

```
net = Network();
net.add(FCLayer((1,2), (1,3)));
net.add(ActivationLayer((1,3), tanth, tanh_prime));
net.add(FCLay((1,3), (1,1)));
net.add(ActivationLayer((1,1), tanth, tanh_prime));
```

3-layer neural network



I’m assuming you already have *some* knowledge about neural networks. The purpose here is not to explain why we make these models, but to show **how to make a proper implementation**.

> 假设你对神经网络已经有一定的了解，这篇文章的目的不是解释为什么构建这些模型，而是要说明**如何正确实现**。

### Layer by Layer  **逐层**

We need to keep in mind the big picture here :

1. We feed **input** data into the neural network.
2. The data flows **from layer to layer** until we have the **output**.
3. Once we have the output, we can calculate the **error** which is a **scalar**.
4. Finally we can adjust a given parameter (weight or bias) by subtracting the **derivative** of the error with respect to the parameter itself.
5. We iterate through that process.

>我们这里需要牢记整个框架：
>\1.    将数据**输入**神经网络
>\2.    在得出输出之前，数据**从一层流向下一层**。
>\3.    一旦得到输出，就可以计算出一个**标量误差**。
>\4.    最后，可以通过相对于参数本身减去误差的**导数**来调整给定参数（权重或偏差）。
>\5.    遍历整个过程。

The most important step is the **4th**. We want to be able to have as many layers as we want, and of any type. But if we modify/add/remove one layer from the network, the output of the network is going to change, which is going to change the error, which is going to change the derivative of the error with respect to the parameters. We need to be able to compute the derivatives regardless of the network architecture, regardless of the activation functions, regardless of the loss we use.

In order to achieve that, we must implement **each layer separately**.

>最重要的一步是**第四步**。 我们希望能够拥有任意数量的层，以及任何类型的层。 但是如果修改/添加/删除网络中的一个层，网络的输出将会改变，误差也将改变，误差相对于参数的导数也将改变。无论网络架构如何、激活函数如何、损失如何，都必须要能够计算导数。
>
>为了实现这一点，我们必须分别**实现每一层**。

### What every layer should implement   **每个层应该实现什么**

Every layer that we might create (fully connected, convolutional, maxpooling, dropout, etc.) have at least 2 things in common : **input** and **output** data.

> 我们可能构建的每一层（完全连接，卷积，最大化，丢失等）至少有两个共同点：输入和输出数据。



![img](.\Pic\2.png)

#### **Now the important part**        **现在重要的一部分**

Suppose that we give a layer the **derivative of the error with respect to its output** (∂E/∂Y), then it must be able to provide the **derivative of the error with respect to its input** (∂E/∂X).

> 假设给出一个层相对于其输出（∂E/∂Y）误差的导数，那么它必须能够提供相对于其输入（∂E/∂X）误差的导数。



![img](.\Pic\3.png)

Remember that `E` is a **scalar** (a number) and `X` and `Y` are **matrices**.



![img](.\Pic\4.png)

We can easily calculate the elements of ∂E/∂X using the chain rule :

> 我们可以使用链规则轻松计算∂E/∂X的元素：



![img](.\Pic\5.png)

#### Why ∂E/∂X ?      **为什么是∂E/∂X？**

For each layer we need the derivative of the error with respect to its **input**because its going to be the derivative of the error with respect to the **previous layer’s output**. This is **very** important, it’s the **key** to understand backpropagation ! After that, we’ll be able to code a Deep Convolutional Neural Network from scratch in no time !

> 对于每一层，我们需要相对于其输入的误差导数，因为它将是相对于前一层输出的误差导数。这非常重要，这是理解反向传播的关键！在这之后，我们将能够立即从头开始编写深度卷积神经网络！

#### Fancy diagrams        **花样图解**

Essentially, for forward propagation, we give the input data to the first layer, then the output of every layer becomes the input of the next layer until we reach the end of the network.

> 基本上，对于前向传播，我们将输入数据提供给第一层，然后每层的输出成为下一层的输入，直到到达网络的末端。

![](.\Pic\6.png)



For backward propagation, we are simply using the chain rule to get the derivatives we need. This is why every layer must provide the derivative of its output with respect to its input.

> 对于反向传播，我们只是简单使用链规则来获得需要的导数。这就是为什么每一层必须提供其输出相对于其输入的导数。



![img](.\Pic\7.png)

This may seem abstract here, but it will get very clear when we will apply this to a specific type of layer. Speaking of *abstract*, now is a good time to write our first python class.

> 这可能看起来很抽象，但是当我们将其应用于特定类型的层时，它将变得非常清楚。现在是编写第一个python类的好时机。

#### Abstract Base Class : Layer          **抽象基类：Layer**

The abstract class *Layer*, which all other layers will inherit from, handles simple properties which are an **input**, an **output**, and both a **forward** and **backward** methods.

> 所有其它层将继承的抽象类Layer会处理简单属性，这些属性是**输入，输出**以及**前向**和**反向**方法。

```python
from abc import abstractmethod

# Base class
class Layer:
    def __init__(self):
        self.input = None;
        self.output = None;
        self.input_shape = None;
        self.output_shape = None;

    # computes the output Y of a layer for a given input X
    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
```

As you can see there is an extra parameter in `backward_propagation` that I didn’t mention, it is the `learning_rate`. This parameter should be something like an update policy, or an optimizer as they call it in Keras, but for the sake of simplicity we’re simply going to pass a learning rate and update our parameters using gradient descent.

> 正如你所看到的，在 back_propagation 函数中，有一个我没有提到的参数，它是 learning_rate 。 此参数应该类似于更新策略或者在Keras中调用它的优化器，为了简单起见，我们只是通过学习率并使用梯度下降更新我们的参数。

### Fully Connected Layer     **全连接层**

Now lets define and implement the first type of layer : fully connected layer or FC layer. FC layers are the most basic layers as every input neurons are connected to every output neurons.

> 现在先定义并实现第一种类型的网络层：全连接层或FC层。FC层是最基本的网络层，因为每个输入神经元都连接到每个输出神经元。



![img](.\Pic\8.png)

#### Forward Propagation    **前向传播**

The value of each output neuron can be calculated as the following :

> 每个输出神经元的值由下式计算：

![](.\Pic\9.png)



With matrices, we can compute this formula for every output neuron in one shot using a **dot product** :

> 使用矩阵，可以使用点积来计算每一个输出神经元的值：



![img](.\Pic\10.png)





We’re done with the forward pass. Now let’s do the backward pass of the FC layer.

> 当完成前向传播之后，现在开始做反向传播。

------

*Note that I’m not using any activation function yet, that’s because we will implement it in a separate layer !*

------

#### Backward Propagation     **反向传播**

As we said, suppose we have a matrix containing the derivative of the error with respect to **that layer’s output** (∂E/∂Y). We need :

> 正如我们所说，假设我们有一个矩阵，其中包含与该层输出相关的误差导数（∂E/∂Y）。 我们需要 ：

1. The derivative of the error with respect to the parameters (∂E/∂W, ∂E/∂B)
2. The derivative of the error with respect to the input (∂E/∂X)



> 1.关于参数的误差导数（∂E/∂W，∂E/∂B）
>
> 2.关于输入的误差导数（∂E/∂X）

Lets calculate ∂E/∂W. This matrix should be the same size as W itself : `ixj`where `i` is the number of input neurons and `j` the number of output neurons. We need **one gradient for every weight** :

>首先计算∂E/∂W，该矩阵应与W本身的大小相同：对于ixj，其中i是输入神经元的数量，j是输出神经元的数量。每个**权重都需要一个梯度**：



![img](.\Pic\11.png)

Using the chain rule stated earlier, we can write :



![img](.\Pic\12.png)

Therefore,



![img](.\Pic\13.png)

That’s it we have the first formula to update the weights ! Now lets calculate ∂E/∂B.

![](.\Pic\14.png)



Again ∂E/∂B needs to be of the same size as B itself, one gradient per bias. We can use the chain rule again :



![img](.\Pic\15.png)

And conclude that,



![img](.\Pic\16.png)

Now that we have **∂E/∂W** and **∂E/∂B**, we are left with **∂E/∂X** which is **very important** as it will “act” as ∂E/∂Y for the layer before that one.

![](.\Pic\17.png)



Again, using the chain rule,



![img](.\Pic\18.png)

Finally, we can write the whole matrix :



![img](.\Pic\19.png)

That’s it ! We have the three formulas we needed for the FC layer !



![img](C:\Users\liumian\Documents\codes\codes\00GitHub\Medium-Python-Neural-Network\Pic\20.png)

#### Coding the Fully Connected Layer   **编码全连接层**

We can now write some python code to bring this math to life !

```python
from layer import Layer
import numpy as np

# inherit from base class Layer
class FCLayer(Layer):
    # input_shape = (1,i)   i the number of input neurons
    # output_shape = (1,j)  j the number of output neurons
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape;
        self.output_shape = output_shape;
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5;
        self.bias = np.random.rand(1, output_shape[1]) - 0.5;

    # returns output for a given input
    def forward_propagation(self, input):
        self.input = input;
        self.output = np.dot(self.input, self.weights) + self.bias;
        return self.output;

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T);
        dWeights = np.dot(self.input.T, output_error);
        # dBias = output_error
        
        # update parameters
        self.weights -= learning_rate * dWeights;
        self.bias -= learning_rate * output_error;
        return input_error;
```



### Activation Layer      **激活层**

All the calculation we did until now were completely linear. Its hopeless to learn anything with that kind of model. We need to add **non-linearity** to the model by applying non linear functions to the output of some layers.

Now we need to redo the whole process for this new type of layer !

No worries, it’s going to be way faster as there are no *learnable* parameters. We just need to calculate **∂E/∂X**.

>到目前为止所做的计算都完全是线性的。用这种模型学习是没有希望的，需要通过将非线性函数应用于某些层的输出来为模型添加非线性。
>
>现在我们需要为这种新类型的层（激活层）重做整个过程！
>
>不用担心，因为此时没有可学习的参数，过程会快点，只需要计算∂E/∂X。

We will call `f` and `f'` the activation function and its derivative respectively.



![img](.\Pic\21.png)

#### Forward Propagation  **前向传播**

As you will see, it is quite straightforward. For a given input `X` , the output is simply the activation function applied to every element of `X` . Which means **input** and **output** have the **same dimensions**.

> 正如将看到的，它非常简单。对于给定的输入X，输出是关于每个X元素的激活函数，这意味着输入和输出具有相同的大小。



![img](.\Pic\22.png)

#### Backward Propagation    **反向传播**

Given **∂E/∂Y**, we want to calculate **∂E/∂X**.



![img](.\Pic\23.png)

Be careful, here we are using an **element-wise** multiplication between the two matrices (whereas in the formulas above, it was a dot product).

#### Coding the Activation Layer        **编码实现激活层**

The code for the activation layer is as straightforward.

```python
from layer import Layer

# inherit from base class Layer
class ActivationLayer(Layer):
    # input_shape = (1,i)   i the number of input neurons
    def __init__(self, input_shape, activation, activation_prime):
        self.input_shape = input_shape;
        self.output_shape = input_shape;
        self.activation = activation;
        self.activation_prime = activation_prime;

    # returns the activated input
    def forward_propagation(self, input):
        self.input = input;
        self.output = self.activation(self.input);
        return self.output;

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error;
```



You can also write some activation functions and their derivatives in a separate file. These will be used later to create an `ActivationLayer`.

```python
import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;
```


### Loss Function      **损失函数**

Until now, for a given layer, we supposed that **∂E/∂Y** was given (by the next layer). But what happens to the last layer ? How does it get **∂E/∂Y** ? We simply give it manually, and it depends on how we define the error.

The error of the network, which measures how good or bad the network did for a given input data, is defined by **you**. There are many ways to define the error, and one of the most known is called **MSE — Mean Squared Error**.

>到目前为止，对于给定的层，我们假设给出了∂E/∂Y（由下一层给出）。但是最后一层怎么得到∂E/∂Y？我们通过简单地手动给出最后一层的∂E/∂Y，它取决于我们如何定义误差。
>
>网络的误差由自己定义，该误差衡量网络对给定输入数据的好坏程度。有许多方法可以定义误差，其中一种最常见的叫做MSE - Mean Squared Error：



![img](.\Pic\24.png)



Where `y*` and `y` denotes **desired output** and **actual output** respectively. You can think of the loss as a last layer which takes all the output neurons and squashes them into one single neuron. What we need now, as for every other layer, is to define **∂E/∂Y**. Except now, we finally reached `E` !

> 其中y *和y分别表示期望的输出和实际输出。你可以将损失视为最后一层，它将所有输出神经元吸收并将它们压成一个神经元。与其他每一层一样，需要定义∂E/∂Y。除了现在，我们终于得到E！



![img](.\Pic\25.png)

These are simply two python functions that you can put in a separate file. They will be used when creating the network.

```python
import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;
```





### Network Class    **网络类**

Almost done ! We are going to make a `Network` class to create neural networks very easily akin the first picture !

I commented almost every part of the code, it shouldn’t be too complicated to understand if you grasped the previous steps. Nevertheless, leave a comment if you have any question, I will gladly answer !

>到现在几乎完成了！我们将构建一个Network类来创建神经网络，非常容易，类似于第一张图片！
>
>我注释了代码的每一部分，如果你掌握了前面的步骤，那么理解它应该不会太复杂。

```python
from layer import Layer

class Network:
    def __init__(self):
        self.layers = [];
        self.loss = None;
        self.loss_prime = None;

    # add layer to network
    def add(self, layer):
        self.layers.append(layer);

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss;
        self.loss_prime = loss_prime;

    # predict output for given input
    def predict(self, input):
        # sample dimension first
        samples = len(input);
        result = [];

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input[i];
            for layer in self.layers:
                # output of layer l is input of layer l+1
                output = layer.forward_propagation(output);
            result.append(output);

        return result;

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train);

        # training loop
        for i in range(epochs):
            err = 0;
            for j in range(samples):
                # forward propagation
                output = x_train[j];
                for layer in self.layers:
                    output = layer.forward_propagation(output);

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output);

                # backward propagation
                error = self.loss_prime(y_train[j], output);
                # loop from end of network to beginning
                for layer in reversed(self.layers):
                    # backpropagate dE
                    error = layer.backward_propagation(error, learning_rate);

            # calculate average error on all samples
            err /= samples;
            print('epoch %d/%d   error=%f' % (i+1,epochs,err));
```



### Building a Neural Network     **构建一个神经网络**

Finally ! We can use our class to create a neural network with as many layers as we want ! For the sake of simplicity, I’m just going to show you how to make… a **XOR**.

> 最后！我们可以使用我们的类来创建一个包含任意数量层的神经网络！为了简单起见，我将向你展示如何构建......一个XOR。

```python
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from losses import *
from activations import *
import numpy as np

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]]);
y_train = np.array([[[0]], [[1]], [[1]], [[0]]]);

# network
net = Network();
net.add(FCLayer((1,2), (1,3)));
net.add(ActivationLayer((1,3), tanh, tanh_prime));
net.add(FCLayer((1,3), (1,1)));
net.add(ActivationLayer((1,1), tanh, tanh_prime));

# train
net.use(mse, mse_prime);
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1);

# test
out = net.predict(x_train);
print(out);
```



Again, I don’t think I need to emphasize many things. Just be careful with the training data, you should always have the **sample** dimension **first**. For example, with the xor problem, the shape should be (4,1,2).

> 同样，我认为不需要强调很多事情，只需要仔细训练数据，应该能够先获得样本维度。例如，对于xor问题，样式应为（4,1,2）。

#### Result

```
$ python xor.py 
epoch 1/1000 error=0.322980
epoch 2/1000 error=0.311174
epoch 3/1000 error=0.307195
...
epoch 998/1000 error=0.000243
epoch 999/1000 error=0.000242
epoch 1000/1000 error=0.000242
[array([[ 0.00077435]]), array([[ 0.97760742]]), array([[ 0.97847793]]), array([[-0.00131305]])]
```

### Convolutional Layer     **卷积层**

This post is starting to be pretty long so I won’t describe all the steps to implement a convolutional layer. However, here’s an implementation that I made :

> 这篇文章开始很长，所以我不会描述实现卷积层的所有步骤。但是，这是我做的一个实现：

```python
from layer import Layer
from scipy import signal
import numpy as np

# inherit from base class Layer
# This convolutional layer is always with stride 1
class ConvLayer(Layer):
    # input_shape = (i,j,d)
    # kernel_shape = (m,n)
    # layer_depth = output depth
    def __init__(self, input_shape, kernel_shape, layer_depth):
        self.input_shape = input_shape;
        self.input_depth = input_shape[2];
        self.kernel_shape = kernel_shape;
        self.layer_depth = layer_depth;
        self.output_shape = (input_shape[0]-kernel_shape[0]+1, input_shape[1]-kernel_shape[1]+1, layer_depth);
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth) - 0.5;
        self.bias = np.random.rand(layer_depth) - 0.5;

    # returns output for a given input
    def forward_propagation(self, input):
        self.input = input;
        self.output = np.zeros(self.output_shape);

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:,:,k] += signal.correlate2d(self.input[:,:,d], self.weights[:,:,d,k], 'valid') + self.bias[k];

        return self.output;

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        in_error = np.zeros(self.input_shape);
        dWeights = np.zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.layer_depth));
        dBias = np.zeros(self.layer_depth);

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:,:,d] += signal.convolve2d(output_error[:,:,k], self.weights[:,:,d,k], 'full');
                dWeights[:,:,d,k] = signal.correlate2d(self.input[:,:,d], output_error[:,:,k], 'valid');
            dBias[k] = self.layer_depth * np.sum(output_error[:,:,k]);

        self.weights -= learning_rate*dWeights;
        self.bias -= learning_rate*dBias;
        return in_error;
```





The math behind it is actually not very complicated ! Here is an excellent post where you’ll find explanations and calculations for **∂E/∂W, ∂E/∂B** and **∂E/∂X**.

>它背后的数学实际上并不复杂！这是一篇很好的文章，你可以找到∂E/∂W，∂E/∂B和∂E/∂X的解释和计算。
>
>如果你想验证你的理解是否正确，请尝试自己实现一些网络层，如MaxPooling，Flatten或Dropout

[**Forward And Backpropagation in Convolutional Neural Network.**
*The below post demonstrates the use of convolution operation for carrying out the back propagation in a CNN.*medium.com](https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e)

If you’d like to check your understanding, try to implement some layers for yourself like MaxPooling, Flatten, or Dropout.

### GitHub Repository

You can find the whole working code used for this post on the following GitHub repository.

[**OmarAflak/Medium-Python-Neural-Network**
*Contribute to OmarAflak/Medium-Python-Neural-Network development by creating an account on GitHub.*github.com](https://github.com/OmarAflak/Medium-Python-Neural-Network)

------

### **If you liked this post — I’d really appreciate if you hit the clap button** 👏 **it would help me a lot. Peace! 😎**