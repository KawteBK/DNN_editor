#girl, woman, person, WE NEED TO ADDA FLATTEN LAYER, fuuuuck, and i thought i was done adding layer
class Layer:

    #i removed the input and output shape from the construction cuz they'll be calculated automatically based on the previous layer and the layer type
    def __init__(self,id, name, role = None ) -> None:

        self._id=id
        self._name=name
        self._role=role
        self._inputShape=None
        self._outputShape=None
        self._isInput = False
        self._isLocked = False
    
    def getOutputShape(self, inputShape): 
        
        raise NotImplementedError("Subclasses must implement getOutputShape()")
   
   
    



class InputLayer(Layer):
    def __init__(self, inputShape):
        super().__init__(id="input", name="Input", role="input")
        self._outputShape = inputShape   # known from the start, set directly
        self._isInput = True
        self._isLocked = True



class Dense(Layer):
    def __init__(self, unit, useBias, dropout,id, name, role) -> None:
        super().__init__(id, name, role)
        self._unit=unit
        self._useBias=useBias 
        self._dropout=dropout

    def getOutputShape(self, inputShape):
        if len(inputShape) != 1:
            raise ValueError(
                f"Dense expects 1D input e.g. (512,) but received {inputShape} — add a Flatten layer first"
            )
        return (self._unit,)









class Conv2D(Layer):
    def __init__(self, filters, kernelSize, strides, padding, dropout, dropoutStyle,id, name, role) -> None:
        super().__init__(id, name, role)
        self._filters=filters
        self._kernelSize=kernelSize
        self._strides=strides
        self._padding=padding
        self._dropout=dropout
        self._dropoutStyle=dropoutStyle
    
    
    def getOutputShape(self, inputShape): 
        
        if len(inputShape) != 3:
            raise ValueError("Conv2D expects input shape (C, H, W)")

        C, H, W = inputShape
        # the output shape should be something like this (filters, H_out, W_out)
              
        H_out = int((H - self._kernelSize[0] + 2 * self._padding[0]) / self._strides[0]) + 1
        W_out = int((W - self._kernelSize[1] + 2 * self._padding[1]) / self._strides[1]) + 1

        return self._filters, H_out, W_out

    









class Recurrent(Layer):
    def __init__(self,hiddenSize , numLayer,dropout,id, name, role, bidirection, returnSeq) -> None:
        super().__init__(id, name, role)      #numLayer is the number of layers in the reccurent block (for example if you have 2 LSTM layers in the block numLayer=2)
        self._hiddenSize=hiddenSize
        self._numLayer=numLayer
        self._dropout=dropout
        self._bidirection=bidirection
        self._returnSeq=returnSeq
       

    def getOutputShape(self, inputShape): 


        # core input shape shoule be something like this: (seq_len, features) and if biderectional  then the hideenSize*2 
        if(len(inputShape)!=2):
           raise ValueError("Recurrent expects input shape (seq_len, features)")
        else:
            seq_len, features = inputShape 
            hiddenSize = self._hiddenSize * 2 if self._bidirection else self._hiddenSize
            if self._returnSeq:
                return (seq_len, hiddenSize)
            else:
                return (hiddenSize,)
        
        
        









class Pooling(Layer): #this will handle only 2D pooling for now, we will add 1D and 3D pooling later, if we have time
    def __init__(self, poolSize, strides, padding,id, name, role) -> None:
        super().__init__(id, name, role)
        self._poolSize=poolSize
        self._strides=strides
        self._padding=padding

#the inputshape should be (C, H, W) and the output shape should be (C, H_out, W_out)
    def getOutputShape(self, inputShape):
        if len(inputShape) != 3:
         raise ValueError("Pooling expects input shape (C, H, W)")
        C,H,W = inputShape
        H_out = int((H - self._poolSize[0] + 2 * self._padding[0]) / self._strides[0]) + 1
        W_out = int((W - self._poolSize[1] + 2 * self._padding[1]) / self._strides[1]) + 1
        return (C,H_out, W_out)
        
     





from enum import Enum

class ActivationType(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LINEAR="linear"
class Activation(Layer): 
    def __init__(self, activationType, id, name, role=None):
        super().__init__(id, name, role)

        self._activationType = activationType

    def getOutputShape(self, inputShape):
        return inputShape