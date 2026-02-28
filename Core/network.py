from layer import Layer
from connection import Connection
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

    def propagate_shapes(self): #this is to calculate the output shape of each layer based on the input shape and the type of the layer, we need to do this before we can do any training or inference, because we need to know the shape of the data that will be passed through the network
        ordered = self.topological_sort()

        for layer in ordered:
            if layer._isInput:
                continue

            parents = self.get_parents(layer._id)

            if len(parents) != 1:
                raise Exception(f"Layer '{layer._id}' has {len(parents)} parents — only single-input supported for now")

            parent_layer = self.layers[parents[0]]

            if parent_layer._outputShape is None:
                raise Exception(f"Parent layer '{parent_layer._id}' has undefined output shape")

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