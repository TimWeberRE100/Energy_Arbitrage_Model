class general_systems:
    def __init__(self, assumptions, linear_assumptions):
        # Technical parameters initialised
        self.obj_type = "general"
        self.type = assumptions["system_type"] # Type of storage system (e.g. PHS, BESS)
        self.energy_capacity = int(assumptions["energy_capacity"]) # Nameplate energy capacity of the system [MWh]
        self.power_capacity = int(assumptions["power_capacity"]) # Nameplate power capacity of the system [MW]
        self.P_min = int(assumptions["P_min"]) # Minimum allowable power output by the system [MW]
        self.P_max = int(assumptions["P_max"]) # Maximum allowable power output by the system [MW]
        self.SOC_min = float(assumptions["SOC_min"]) # Minimum allowable SOC of the system [0,1]
        self.SOC_max = float(assumptions["SOC_max_initial"]) # Maximum allowable SOC of the system [0,1]
        self.lifetime = int(assumptions["Lifetime"]) # Expected economic life of the system [years]
        self.SOC_initial = float(assumptions["SOC_initial"])

        # Distributable unit parameters initialised
        self.mlf_load = float(assumptions["mlf_load"]) # Transmission loss factor for the load (0,2)
        self.mlf_gen = float(assumptions["mlf_gen"]) # Transmission loss factor for the generator (0,2)
        self.dlf_load = float(assumptions["dlf_load"]) # Distribution loss factor for the load (0,2)
        self.dlf_gen = float(assumptions["dlf_gen"]) # Distribution loss factor for the generator (0,2)

        # Cost parameters initialised
        self.FOM = float(assumptions["FOM"]) # Fixed operation and maintenance costs [$/kW-year]
        self.VOMd = float(assumptions["VOMd"]) # Variable operation and maintenance cost for discharging [$/MWh]
        self.VOMc = float(assumptions["VOMc"]) # Variable operation and maintenance cost for charging [$/MWh]
        self.OCC_e = float(assumptions["OvernightCapitalCost (energy)"]) # Overnight capital cost based on energy capacity [$/MWh]
        self.OCC_p = float(assumptions["OvernightCapitalCost (power)"]) # Overnight capital cost based on power capacity [$/MW]
        self.OCC_f = float(assumptions["OtherOCC"]) # Fixed capital cost, independent of energy and power capacity [$]
        self.discountRate = float(assumptions["discountRate"]) # Discount rate, assumed to be weighted cost of capital (0,1]
        self.i_credit = float(assumptions["i"]) # Investment tax credit (0,1]
        self.corp_tax = float(assumptions["alpha"]) # Effective corporate tax rate (0,1]

        # Linearisation parameters initialised
        self.a = self.linearParameterDF(linear_assumptions, "a")
        self.b = self.linearParameterDF(linear_assumptions, "b")
        self.c = self.linearParameterDF(linear_assumptions, "c")
        self.d = self.linearParameterDF(linear_assumptions, "d")
        self.sdLoss = self.linearParameterDF(linear_assumptions, "sdLoss")
        self.scLoss = self.linearParameterDF(linear_assumptions, "scLoss")

        # Current state parameters initialised
        self.cycle_tracker = 0 # Tracks the current cycle state of the system [0 or 1]
        self.SOC_current = 0.5 # Current SOC of the system [0,1]
        self.P_current = 0 # Current power output by the system [MW]
        self.SOC_pre = 0.5 # Previous SOC of the system [0,1]
        self.P_pre = 0 # Previous power output by the system [MW]
        self.behaviour = 0 # Current charging behaviour of the system: 0 is idle, -1 is discharging, 1 is charging

    def linearParameterDF(self, linearisation_df, parameterName):
        '''
        Builds a dataframe for a particular linearisation parameter
        
        Parameters
        ----------
        linearisation_df : DataFrame
            Dataframe of all linearisation parameters.
        parameterName : string
            The name of the particular linearisation parameter within the linearisation_df.

        Returns
        -------
        parameterValue_df : DataFrame
            Dataframe containing the values for the particular linearisation parameter.

        '''
        parameterValue_df = linearisation_df.loc[(linearisation_df['Variable Name'] == parameterName) & (linearisation_df['System Type'] == self.type)]['Variable Value'].values   
        return parameterValue_df

    def testToCurrent(self):        
        self.SOC_pre = self.SOC_current
        self.P_pre = self.P_current
    
    def idleInterval(self):
        self.P_pre = 0

    def updateSOC_current(self, SOC_exp):
        self.SOC_current = SOC_exp

    def updateP_current(self, P_exp):
        self.P_current = P_exp

    def updateCycleTracker(self, value):
        self.cycle_tracker = value