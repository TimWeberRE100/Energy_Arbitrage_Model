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
        self.SP = [] # List of spot prices for the year [$/MWh]
        self.dailyCycles = [] # List of the number of cycles per day
        self.data = []

class memory_daily:
    def __init__(self, storage_system_inst):
        self.SOC_day = [storage_system_inst.SOC_current]

        if storage_system_inst.P_current < 0:
            self.chargingCapacity = [-storage_system_inst.P_current]
            self.dischargingCapacity = [0]
        else:
            self.dischargingCapacity = [storage_system_inst.P_current]
            self.chargingCapacity = [0]

        self.behaviour = [] # charge = -1, standby = 0, discharge = 1
        self.headLossPump = []
        self.headLossTurbine = []
        self.headPump = []
        self.headTurbine = []
        self.flowRatePump = []
        self.flowRateTurbine = []
        self.efficiencyTurbineDay = []
        self.dischargedEnergy = []
        self.chargedEnergy = []
        self.U_batt_day = []
        self.eff_volt_day = []
        self.calendarLossDay = []
        self.cycleLossDay = []
        self.R_cell_day = []
        self.SOC_max_day = []
