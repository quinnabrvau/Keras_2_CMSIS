## Keras -> CMSIS
This program provides the functionality to convert Keras machine learning models (and saved .h5 models) into CMSIS NN/DSP C files


#### Currently implemented 
* 1d CNN 

#### ToDo
* fully connected NN
* pooling
* quantization optimization 
* auto detect when to quantize to q15 instead of q7


#### Notes
This is a port from an older project that implemented a custom floating point implementation of the 1d cnn functions.
As a result, this project may have some kinks in it while I try and implement the missing functionality.
