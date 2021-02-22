# Convolutional-Neural-Networks-from-Scratch
### Python Library for creating and training CNNs. Implemented from scratch.


This repo contains a project i did during my second year in college.
I wanted to have a deeper understanding of how gradients were calculated and backpropagated through the Network and I felt like this would be a good problem to test my skills on.

The project can be divided into 4 parts:

* ##### **Optimizers.py**
* ##### **ConvNetModule.py**
* ##### **Layers.py**



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

There are still somethings that need work and can be found [here](https://github.com/Cabbagito/Convolutional-Neural-Networks-from-Scratch/issues "Issues").
