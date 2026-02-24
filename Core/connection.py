class Connection:
    def __init__(self,id :str, fromLayer:str, toLayer:str ) -> None:
        self._id=id
        self._fromLayer=fromLayer
        self._toLayer=toLayer
    
    def isValid(self):
        if self._fromLayer is None or self._toLayer is None:
            return False
        return True
        