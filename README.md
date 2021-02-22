# Convolutional-Neural-Networks-from-Scratch
### Python Library for creating and training CNNs. Implemented from scratch.


This repo contains a project i did during my second year in college.
I wanted to have a deeper understanding of how gradients were calculated and backpropagated through the Network and I felt like if I could implement backprop of a Convolutional Nerual Net that would help me immensely.<br/> <br/>
The project can be divided into 3 parts:

* ##### **Optimizers.py**
* ##### **ConvNetModule.py**
* ##### **Layers.py**


<br/>

___
### **Optimizers.py**
Contains four Algorithms for optimization
* SGD
* Adam
* Momentum
* RMSProp

___
### **ConvNetModule.py**
Controls the training process, interacts with the layers and sends the gradients to the optimizers.

___
### **Layers.py**
The hardest part to implement. Contains forward and backward operations for all the layers.

___
Additional Libraries used: Numpy(matrix operations), tqdm(appearance) and Matplotlib(plotting).
___

There are still some things that need work and can be found [here](https://github.com/Cabbagito/Convolutional-Neural-Networks-from-Scratch/issues "Issues").

For help with understanding the Backward operation of Convolutions: <br/>
https://github.com/JeyrajK/convolutional-neural-networks/blob/master/Back%20propagation%20of%20cnn.ipynb <br/>
https://towardsdatascience.com/forward-and-backward-propagations-for-2d-convolutional-layers-ed970f8bf602 <br/>
https://cs231n.github.io/ <br/>
https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/ <br/>
https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
