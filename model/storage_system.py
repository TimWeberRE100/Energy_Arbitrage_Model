'''Define the constructor class for the storage system.'''

class storage_system:
    def __init__(self, *args):
        '''Construct the storage_system object from the args objects.
        
        Parameters
        ----------
        args : Any
            Should specify a general_systems object and a specific storage technology object (e.g. battery)
        '''
        self.__objects = args

    def __getattr__(self, name):
        for obj in self.__objects:
            try:
                return getattr(obj, name)
            except AttributeError:
                pass

        raise AttributeError

    def testToCurrent(self):
        '''Update the current state for the previous dispatch interval if the tested state is accepted.'''
        for obj in self._storage_system__objects:
            obj.testToCurrent()

    def idleInterval(self):
        '''Update the current state for the previous dispatch interval if the tested state is rejected.'''
        for obj in self._storage_system__objects:
            obj.idleInterval()
        
