from .layer import Layer
from connection import Connection
class Network:
    def __init__(self, id,):
        self.id = id
        self.layers : dict[str,Layer] = {}      #this should be a dict (it was a map in java <id,Layer>)
        self.connections : list[Connection] = []    # this should be a list of connections

    def addLayer(self, layer: Layer):
        if layer._id in self.layers:
            return False

        self.layers[layer._id] = layer # add the layer to the dictionary using its id as the key
        return True
    

    
    def removeLayer(self, layerId: str):
        if layerId not in self.layers:
            return False
        
        if self.layers[layerId]._isLocked:
            return False
        
        del self.layers[layerId]

        #the layer is removed BUT we need to remove all conncection with this layer as well
        #sorry connections, gotta do what i need to do 
        self.connections = [
            conn for conn in self.connections
            if conn._fromLayer != layerId and conn._toLayer != layerId   
            ] #the first conn will take the value of the connection that will get after itirating on all connection and finds its 'to' and 'from' not equal to the layer we just deleted
        return True
    



    def addConnect(self, fromLayerId: str, toLayerId: str):
        if fromLayerId not in self.layers or toLayerId not in self.layers:
            return False
        if fromLayerId == toLayerId: # no fucker is gonna connect the same layer to itself 
            return False
        
        if any(conn._fromLayer == fromLayerId and conn._toLayer == toLayerId for conn in self.connections): #means the connection already exists
            return False
        
        if self.layers[toLayerId]._isInput: #we ain't making connection to the input layer
            return False
        
        
        connection = Connection(fromLayerId, toLayerId) #we create connection, check if it make a cycle; if yes, then we remove it if not, we keep it
        self.connections.append(connection)

        try:
           self.topological_sort()
        except Exception:
           self.connections.pop()   # rollback, no cycles in this household
           return False
        
        return True
    

    

    def removeConnect(self, fromLayerId: str, toLayerId: str):
        for i, conn in enumerate(self.connections):
            if conn._fromLayer == fromLayerId and conn._toLayer == toLayerId:
                del self.connections[i]
                return True
        return False
    

    def getInputLayers(self) -> list[Layer]:
        incoming = {conn._toLayer for conn in self.connections}

        return [
            layer for layer_id, layer in self.layers.items()
            if layer_id not in incoming
        ]
    
    def getOutputLayers(self):
        outgoing = {conn._fromLayer for conn in self.connections}

        return [
            layer for layer_id, layer in self.layers.items()
            if layer_id not in outgoing
        ]


    def validate(self): #here we're just cheking if the structure is graph- valid 
        if not self.layers: #empty graph isn't a graph, get the fuck out here
            return False

        try:
            self.topological_sort() #yeah, no cycle allowed you too pack your shit and go
        except:
          return False

        if not self.getInputLayers(): #not every layer should have a parent!!! input shouldn't have a parent
             return False

        if not self.getOutputLayers(): #there must be at last one output layer, again no fucking cycles in here
             return False

        return True
    
    def generateSummary(self) -> dict[str, str]:
        return {
            "Network ID": self.id,
            "Total Layers": str(len(self.layers)),
            "Total Connections": str(len(self.connections)),
            "Input Layers": ", ".join(l._id for l in self.getInputLayers()),
            "Output Layers": ", ".join(l._id for l in self.getOutputLayers())
        }
        

        


      # this one is to find the correct execution order of layers in a directed graph
      # using Kahn’s Algorithm

    def topological_sort(self): #fucking study the kahn algorithm if you want to understand this shit

        from collections import deque

        in_degree = {layer_id: 0 for layer_id in self.layers} #we give every node(layer) 0 as number of edges

        # calculate the incoming edges
        for conn in self.connections:
             in_degree[conn._toLayer] += 1

        queue = deque(
            [layer_id for layer_id, deg in in_degree.items() if deg == 0]
            )

        sorted_layers = []

        while queue:
            current = queue.popleft()
            sorted_layers.append(self.layers[current])

            for conn in self.connections:
              if conn._fromLayer == current:
                 in_degree[conn._toLayer] -= 1
                 if in_degree[conn._toLayer] == 0:
                        queue.append(conn._toLayer)

        if len(sorted_layers) != len(self.layers):
            raise Exception("Cycle detected in network")

        return sorted_layers
    

    def propagate_shapes(self):
     ordered = self.topological_sort()

     for layer in ordered:
             if layer._isInput:
                continue

             parents = self.get_parents(layer._id)

             if len(parents) != 1:
                 raise Exception("Only single-input supported for now")

             parent_layer = self.layers[parents[0]]
             
             layer._outputShape = layer.getOutputShape(parent_layer._outputShape)
             if parent_layer._outputShape is None:
                raise Exception(f"Parent layer {parent_layer._id} has undefined output shape")



    def get_parents(self, layerId):
         return [
        conn._fromLayer
        for conn in self.connections
        if conn._toLayer == layerId
         ]