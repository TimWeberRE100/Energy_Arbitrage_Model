class memory:
    def __init__(self):
        self.TA_dis = [] # List of trading amounts for discharged energy [$]
        self.TA_ch = [] # List of trading amounts for discharged energy [$]
        self.DischargedEnergy = [] # List of discharged energy totals [MWh]
        self.ChargedEnergy = [] # List of charged energy totals [MWh]
        self.capacityFactor = [] # List of capacity factors (0,1]
        self.averageCycleTime = [] # List of average cycle times [cycles per day]
        self.finalSOCmax = [] # List containing the state of health at the end of each period (0,1]
        self.finalRcell = [] # List containing the internal resistance at the end of each period [Ohms]
        self.data = []
