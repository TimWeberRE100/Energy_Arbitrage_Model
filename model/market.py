'''
Define the class for an electricity market.

Classes
-------
market
'''

class market:
    def __init__(self, assumptions):
        self.SPmin = int(assumptions["SPmin"]) # Price floor for the spot market [$/MWh]
        self.SPmax = int(assumptions["SPmax"]) # Price ceiling for the spot market [$/MWh]
        self.dispatch_t = int(assumptions["Dispatch_interval_time"]) # Dispatch interval length [min]
        