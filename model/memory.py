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
        self.dispatch_prices = []
    
    def update_phs(self, behaviour, headLossPump, headPump, flowRatePump, headLossTurbine, headTurbine, flowRateTurbine, efficiencyTurbineDay):
        self.behaviour.append(behaviour)
        self.headLossPump.append(headLossPump)
        self.headPump.append(headPump)
        self.flowRatePump.append(flowRatePump)
        self.headLossTurbine.append(headLossTurbine)
        self.headTurbine.append(headTurbine)
        self.flowRateTurbine.append(flowRateTurbine)
        self.efficiencyTurbineDay.append(efficiencyTurbineDay)

    def update_bess(self, calendarLossDay, cycleLossDay, R_cell_day, SOC_max_day):
        self.calendarLossDay.append(calendarLossDay)
        self.cycleLossDay.append(cycleLossDay)
        self.R_cell_day.append(R_cell_day)
        self.SOC_max_day.append(SOC_max_day)     

    def update_general(self, chargingCapacity, dischargingCapacity, chargedEnergy, dischargedEnergy, SOC_day):
        self.chargingCapacity.append(chargingCapacity)
        self.dischargingCapacity.append(dischargingCapacity)
        self.chargedEnergy.append(chargedEnergy)
        self.dischargedEnergy.append(dischargedEnergy)
        self.SOC_day.append(SOC_day)

