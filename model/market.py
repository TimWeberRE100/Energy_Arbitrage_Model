class market:
    def __init__(self, assumptions):
        self.SPmin = -1000 # Price floor for the spot market [$/MWh]
        self.SPmax = 15500 # Price ceiling for the spot market [$/MWh]
        self.dispatch_t = 5 # Dispatch interval length [min]
        