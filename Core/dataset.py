class Dataset:
    def __init__(self, id:str ,name:str, inputShape: tuple, numClasses: int):
        self._id = id
        self._name = name
        self._inputShape = inputShape
        self._numClasses = numClasses


       # later i want to scale it up to be able to upload datasets , not use just the prebuilt ones
        #up to that time, we'll be needing a load methode and the paths aaaaand attributes : trainpath, testpath, valpath
    def validate(self) -> bool:
        if not self._id or not self._name:
            return False
        if not self._inputShape or len(self._inputShape) == 0:
            return False
        if self._numClasses < 1:
            return False
        return True
       