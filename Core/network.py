from .layer import (Activation, ActivationType, Conv2D, DropoutStyle,Dense, Flatten, InputLayer, Layer,
                   Pooling, PoolingType, Recurrent, RecurrentType)
from .connection import Connection
from collections import deque
class Network:
    def __init__(self, id:str):
        self.id = id
        self.layers : dict[str,Layer] = {}      #this should be a dict (it was a map in java <id,Layer>)
        self.connections : list[Connection] = []    # this should be a list of connections

        self._outgoing: dict[str, list[str]] = {} # layerId —> [children ids]
        self._incoming: dict[str, list [str]] = {}  # layerId —> [parent ids]
         

    def addLayer(self, layer: Layer):
        if layer._id in self.layers:
            return False

        self.layers[layer._id] = layer # add the layer to the dictionary using its id as the key
        self._outgoing[layer._id] = []   # new node starts with no edges
        self._incoming[layer._id] = []
        return True
    

    
    def removeLayer(self, layerId: str):
        if layerId not in self.layers:
            return False
        
        if self.layers[layerId]._isLocked:
            return False
        
        del self.layers[layerId]
        for child in self._outgoing[layerId]:
            self._incoming[child].remove(layerId)

        for parent in self._incoming[layerId]:
            self._outgoing[parent].remove(layerId)


        #the layer is removed BUT we need to remove all conncection with this layer as well
        #sorry connections, gotta do what i need to do 
        self.connections = [
            conn for conn in self.connections
            if conn._fromLayer != layerId and conn._toLayer != layerId   
            ] #the first conn will take the value of the connection that will get after itirating on all connection and finds its 'to' and 'from' not equal to the layer we just deleted
        del self._outgoing[layerId]
        del self._incoming[layerId]
        

        return True
    



    def addConnect(self, fromLayerId: str, toLayerId: str):
        if fromLayerId not in self.layers or toLayerId not in self.layers:
            return False
        if fromLayerId == toLayerId: # no fucker is gonna connect the same layer to itself 
            return False
        
        if toLayerId in self._outgoing[fromLayerId]:   # O(degree) but degree is small
            return False
        
        if self.layers[toLayerId]._isInput: #we ain't making connection to the input layer
            return False
        
        
        self._outgoing[fromLayerId].append(toLayerId)
        self._incoming[toLayerId].append(fromLayerId)

        connection = Connection(f"{fromLayerId}->{toLayerId}", fromLayerId, toLayerId)
        self.connections.append(connection)

        try:
          self.topological_sort()

           # Dry-run shape validation: propagate shapes, then reset any layers that were
           # previously shapeless — so explicit propagate_shapes() stays the authoritative setter.
          shapeless = {lid for lid, layer in self.layers.items()
                        if layer._inputShape is None and layer._outputShape is None and not layer._isInput}
          self.propagate_shapes()
          for lid in shapeless:
               self.layers[lid]._inputShape  = None
               self.layers[lid]._outputShape = None
        except Exception:
            self._outgoing[fromLayerId].remove(toLayerId)
            self._incoming[toLayerId].remove(fromLayerId)
            self.connections.pop()   # rollback, no cycles in this household
            return False
        
        return True
    

    

    def removeConnect(self, fromLayerId: str, toLayerId: str):
        if fromLayerId not in self.layers or toLayerId not in self.layers:
            return False

        if toLayerId not in self._outgoing[fromLayerId]:
            return False  # connection doesn't exist

        self._outgoing[fromLayerId].remove(toLayerId)
        self._incoming[toLayerId].remove(fromLayerId)

        self.connections = [
            conn for conn in self.connections
            if not (conn._fromLayer == fromLayerId and conn._toLayer == toLayerId)
        ]
        return True
    



    def getInputLayers(self) -> list[Layer]:

        return [self.layers[lid] for lid, parents in self._incoming.items() if not parents]

        

        
    
    def getOutputLayers(self):
        return [self.layers[lid] for lid, children in self._outgoing.items() if not children]

    
    def get_parents(self, layerId):
         return self._incoming.get(layerId,[])
    
    def get_children(self,layerId):
        return self._outgoing.get(layerId,[])




        


      # this one is to find the correct execution order of layers in a directed graph
      # using Kahn’s Algorithm

    
    
    def topological_sort(self):

        in_degree = {lid: len(parents) for lid, parents in self._incoming.items()}
        queue = deque([lid for lid, deg in in_degree.items() if deg == 0])
        sorted_layers = []



        while queue:
            current = queue.popleft()
            sorted_layers.append(self.layers[current])

            for child in self._outgoing[current]:   # O(1) lookup, no scanning , hell yeaah
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_layers) != len(self.layers):
            raise Exception("Cycle detected in network")

        return sorted_layers

    def propagate_shapes(self): 
        #this is to calculate the output shape of each layer based on the input shape and the type of the layer, we need to do this before we can do any training or inference, because we need to know the shape of the data that will be passed through the network
        ordered = self.topological_sort()

        for layer in ordered:
            if layer._isInput:
                continue

            parents = self.get_parents(layer._id)
            if len(parents)==0:
                continue
            if len(parents) != 1:
                raise Exception(f"Layer '{layer._id}' has {len(parents)} parents — only single-input supported for now")

            parent_layer = self.layers[parents[0]]

            if parent_layer._outputShape is None:
                raise Exception(f"Parent layer '{parent_layer._id}' has undefined output shape")
            layer._inputShape = parent_layer._outputShape  
            layer._outputShape = layer.getOutputShape(parent_layer._outputShape)




    def validate(self) -> bool:
        if not self.layers:
            return False

        try:
            self.topological_sort()
        except Exception:
            return False

        incoming = self._incoming
        outgoing = self._outgoing

        has_input  = any(not parents  for parents  in incoming.values())
        has_output = any(not children for children in outgoing.values())

        return has_input and has_output


    def generateSummary(self) -> dict[str, str]:
        return {
            "Network ID":         self.id,
            "Total Layers":       str(len(self.layers)),
            "Total Connections":  str(len(self.connections)),
            "Input Layers":       ", ".join(l._id for l in self.getInputLayers()),
            "Output Layers":      ", ".join(l._id for l in self.getOutputLayers()),
        }
    
    
    
    

 #for the deserialization of the network, we need to be able to convert it to a dictionary and back, this is for the project saving and loading, and also for the future when we want to upload networks and need to store their metadata in the project file
    @classmethod
    def fromDict(cls, data: dict) -> "Network":
       
        if not data:
            return None

        network = cls(id=data["id"])
       #becauuuuuse when saveen in json files, the outputshapesare saved as lists, but we need them to be tuples when creating the layers, so we need to convert them back to tuples, 
        def to_tuple(val):
            return tuple(val) if val is not None else None



    # rebuilding layers by type
        for lid, ldata in data["layers"].items():
            layer_type = ldata["type"]

            if layer_type == "InputLayer":
                layer = InputLayer(inputShape=to_tuple(ldata["outputShape"]))

            elif layer_type == "Flatten":
                layer = Flatten(id=ldata["id"], name=ldata["name"], role=ldata["role"])


            elif layer_type == "Dense":
                layer = Dense(unit=ldata["unit"], useBias=ldata["useBias"],
                              dropout=ldata["dropout"], id=ldata["id"],
                              name=ldata["name"], role=ldata["role"])
            

            elif layer_type == "Conv2D":
                raw_ds = ldata.get("dropoutStyle")
                dropout_style = DropoutStyle(raw_ds) if raw_ds else None
                layer = Conv2D(
                    filters=ldata["filters"],
                    kernelSize=tuple(ldata["kernelSize"]),
                    strides=tuple(ldata["strides"]),
                    padding=tuple(ldata["padding"]),
                    dropout=ldata["dropout"],
                    dropoutStyle=dropout_style,
                    id=ldata["id"], name=ldata["name"], role=ldata["role"],
                )

            elif layer_type == "Recurrent":
                layer = Recurrent(
                    hiddenSize=ldata["hiddenSize"],
                    numLayer=ldata["numLayer"],
                    dropout=ldata["dropout"],
                    id=ldata["id"], name=ldata["name"], role=ldata["role"],
                    bidirection=ldata["bidirection"],
                    returnSeq=ldata["returnSeq"],
                    recurrentType=RecurrentType(ldata.get("recurrentType", "lstm")),
                )

            elif layer_type == "Pooling":
                layer = Pooling(
                    poolSize=tuple(ldata["poolSize"]),
                    strides=tuple(ldata["strides"]),
                    padding=tuple(ldata["padding"]),
                    id=ldata["id"], name=ldata["name"], role=ldata["role"],
                    poolingType=PoolingType(ldata.get("poolingType", "max")),
                )

            elif layer_type == "Activation":
                layer = Activation(activationType=ActivationType(ldata["activationType"]),
                                   id=ldata["id"], name=ldata["name"], role=ldata["role"])
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

             # restore saved shapes (skip InputLayer — its shape is set in the constructor)
            if layer_type != "InputLayer":
                layer._inputShape  = to_tuple(ldata.get("inputShape"))
                layer._outputShape = to_tuple(ldata.get("outputShape"))

            network.addLayer(layer)

        for cdata in data["connections"]:
            network.addConnect(cdata["from"], cdata["to"])

        return network
    

    def toDict(self) -> dict:
            return {
            "id": self.id,
        "layers": {lid: self._serializeLayer(layer) for lid, layer in self.layers.items()},
        "connections": [
            {"id": conn._id, "from": conn._fromLayer, "to": conn._toLayer}
            for conn in self.connections
        ]
    }

    def _serializeLayer(self, layer) -> dict:
        base = {
        "id": layer._id,
        "name": layer._name,
        "role": layer._role,
        "type": type(layer).__name__,
        "inputShape": layer._inputShape,
        "outputShape": layer._outputShape,
         }
        if isinstance(layer, InputLayer):
            pass   # no extra fields needed outputShape already in base

        elif isinstance(layer, Flatten):
            pass   # no parameters shape is computed automatically

        elif isinstance(layer, Dense):
            base.update({
                "unit":    layer._unit,
                "useBias": layer._useBias,
                "dropout": layer._dropout,
            })

        elif isinstance(layer, Conv2D):
            base.update({
                "filters":      layer._filters,
                "kernelSize":   layer._kernelSize,
                "strides":      layer._strides,
                "padding":      layer._padding,
                "dropout":      layer._dropout,
                "dropoutStyle": layer._dropoutStyle.value if layer._dropoutStyle else None,
            })

        elif isinstance(layer, Recurrent):
            base.update({
                "hiddenSize":    layer._hiddenSize,
                "numLayer":      layer._numLayer,
                "dropout":       layer._dropout,
                "bidirection":   layer._bidirection,
                "returnSeq":     layer._returnSeq,
                "recurrentType": layer._recurrentType.value,
            })

        elif isinstance(layer, Pooling):
            base.update({
                "poolSize":    layer._poolSize,
                "strides":     layer._strides,
                "padding":     layer._padding,
                "poolingType": layer._poolingType.value,
            })

        elif isinstance(layer, Activation):
            base.update({
                "activationType": layer._activationType.value,
            })

        return base
