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
        del self.layers[layerId]

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
    
    def get_parents(self, layerId):
         return self._incoming.get[layerId, []]


    def validate(self): #here we're just cheking if the structure is graph- valid 
        if not self.layers:
            return False

        try:
            sorted_layers = self.topological_sort()
        except Exception:
            return False

    # reuse connection data you already have
        incoming = {conn._toLayer for conn in self.connections}
        outgoing = {conn._fromLayer for conn in self.connections}

        has_input = any(lid not in incoming for lid in self.layers)
        has_output = any(lid not in outgoing for lid in self.layers)

        return has_input and has_output
    


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

    def topological_sort_oldone(self): #this version is slow, but i am attached to it its like O(v*e)

        from collections import deque
        #bismilah, we initialize the dict, for now y'all have 0 in-degrees
        in_degree = {layer_id: 0 for layer_id in self.layers} #we give every node(layer) 0 as number of edges

        # calculate the incoming edges
        for conn in self.connections: #we iterate on all connection, if we find a tolayeer id, we increment this layer's in-degrees 
             in_degree[conn._toLayer] += 1

        queue = deque(
            [layer_id for layer_id, deg in in_degree.items() if deg == 0] # in_degree.items()  returns tuples like ("id1", 2)
            )

        sorted_layers = []

        while queue: 
            current = queue.popleft() #we remove the first layer (that should have 0 indegrees) from the queue then we add it to the sortedlayers
            sorted_layers.append(self.layers[current]) #cuurent have the id of the layer, and we append that layer, not as an id , like the object layer

            for conn in self.connections:
              if conn._fromLayer == current:    #we find all the nodes that had inconection from the layer we removed from the queue(current)
                 in_degree[conn._toLayer] -= 1      #we decrement the in-degrees of those layers by 1, because we just "removed" one of their incoming edges
                 if in_degree[conn._toLayer] == 0:       #and after the decrease, if it will no longer have any incoming edges (ywli sommet source), we add it to the queue to be processed in the next iterations
                        queue.append(conn._toLayer)

        if len(sorted_layers) != len(self.layers):
            raise Exception("Cycle detected in network!")

        return sorted_layers
    
    
    
    def topological_sort(self):
        from collections import deque, defaultdict

        in_degree = {layer_id: 0 for layer_id in self.layers}
        adjacency = defaultdict(list)

        for conn in self.connections:
            in_degree[conn._toLayer] += 1
            adjacency[conn._fromLayer].append(conn._toLayer) 

        queue = deque([lid for lid, deg in in_degree.items() if deg == 0])
        sorted_layers = []

        while queue:
            current = queue.popleft()
            sorted_layers.append(self.layers[current])
            
            for neighbor in adjacency[current]:  # O(1) lookup instead of scanning all edges
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
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
                 raise Exception("Only single-input supported for now")

             parent_layer = self.layers[parents[0]]
             
             if parent_layer._outputShape is None:
                raise Exception(f"Parent layer {parent_layer._id} has undefined output shape")
             layer._outputShape = layer.getOutputShape(parent_layer._outputShape)




    