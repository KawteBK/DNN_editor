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


    #for the serialization and deserialization of the dataset, we need to be able to convert it to a dictionary and back, this is for the project saving and loading, and also for the future when we want to upload datasets and need to store their metadata in the project file
    def toDict(self) -> dict:
        return {
            "id": self._id,
            "name": self._name,
            "inputShape": list(self._inputShape),
            "numClasses": self._numClasses,
            # "trainPath": self._trainPath,
            # "valPath": self._valPath,
            # "testPath": self._testPath,
        }

    @classmethod
    def fromDict(cls, data: dict) -> "Dataset":
        return cls(
            id=data["id"],
            name=data["name"],
            inputShape=tuple(data["inputShape"]),
            numClasses=data["numClasses"],
            # trainPath=data.get("trainPath"),
            # valPath=data.get("valPath"),
            # testPath=data.get("testPath"),
        )

    
       