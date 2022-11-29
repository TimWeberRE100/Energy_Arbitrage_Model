
class storage_system:
    def __init__(self, *args):
        self.__objects = args

    def __getattr__(self, name):
        for obj in self.__objects:
            try:
                return getattr(obj, name)
            except AttributeError:
                pass

        raise AttributeError

    def testToCurrent(self):
        for obj in self._storage_system__objects:
            obj.testToCurrent()

    def idleInterval(self):
        for obj in self._storage_system__objects:
            obj.idleInterval()
        
