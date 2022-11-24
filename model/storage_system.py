
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
