import numpy as np

def U_OCV_calc(SOC):
        '''
        Calculates the open-circuit voltage of the cells at a particular SOC. Updates the battery's attribute.

        Parameters
        ----------
        SOC : float
            State of charge of the system.

        Returns
        -------
        U_cell : float
            Open circuit voltage of the cell

        '''
        n1 = 5.163*np.exp(-(((SOC-1.794)/1.665)**2))
        n2 = 0.3296*np.exp(-(((SOC-0.6405)/0.3274)**2))
        n3 = 1.59*np.exp(-(((SOC-0.06475)/0.4406)**2))
        n4 = 5.184*np.exp(-(((SOC-(-0.531))/0.3059)**2))
        U_cell = sum([n1,n2,n3,n4])
        return U_cell

class battery:
    def __init__(self, assumptions):
        # Assumed parameters initialised
        self.efficiency_sys = assumptions["efficiency_sys"] # System efficiency of the battery, including HVAC and other auxiliaries (0,1]
        self.P_standby = assumptions["P_standby"] # Power consumption of idling system [MW]
        self.temp = assumptions["Temp"] # Temperature of the battery [K]
        self.eol = assumptions["eol"] # State of health at which point the Li-ion battery is assumed to reach end of life [0,1)
        self.R_cell_initial = assumptions["R_cell_initial"] # Initial internal resistance of the cells at start of simulation [Ohms]
        self.series_cells = assumptions["series_cells"] # Number of cells in series
        self.parallel_cells = assumptions["parallel_cells"] # Number of cells in parallel
        self.cell_e = assumptions["cell_energy_capacity [Ah]"] # Energy capacity of each cell [Ah]
        self.U_cell_nom = U_OCV_calc(0.5) # Nominal open-circuit voltage of the cell [V]
        self.U_batt_nom = self.U_cell_nom * self.series_cells # Nominal open-circuit voltage of the battery [V]
        self.I_cyc = 0 # Current threshold for cycle aging [A]

        # Current state parameters initialised
        self.cycLossCurrentSum = 0 # Sum of filtered current measured during each cycle loss dispatch interval [A]
        self.cycLossIntervals = 0 # Number of dispatch intervals where cycle aging has dominated
        self.calLossCurrentSum = 0 # Sum of all states of charge measured at each dispatch interval where calendar aging has dominated
        self.calLossIntervals = 0 # Number of dispatch intervals where calendar aging has dominated
        self.Ah_throughput = 0 # Energy throughput of the cells over lifetime [Ah]
        self.calLossTime = 0 # Time spent in calendar loss dominated dispatch intervals [h]
        self.SOC_max_loss_cal = 0 # Current reduction in state of health due to calendar aging [0,1]
        self.SOC_max_loss_cyc = 0 # Current reduction in state of health due to cycle ageing [0,1]
        self.SOC_sum = 0.5
        self.U_cell = 0 # Open-circuit voltage of the cell [V]
        self.R_cell = 0 # Internal resistance of the cell [Ohms]
        self.U_batt = 0 # Open-circuit voltage of the battery [V]
        self.efficiency_volt = 1 # Voltage efficiency of the battery (0,1]
        self.I_filt = 0 # Filtered current through the cells [A]

    def U_OCV_assign(self):
        '''
        Calculates the open-circuit voltage of the cells at a particular SOC. Updates the battery's attribute.

        Parameters
        ----------
        SOC : float
            State of charge of the system.

        Returns
        -------
        None

        '''
        self.U_cell = U_OCV_calc(self.SOC_current)
        self.U_batt = self.U_cell * self.series_cells

    def R_cell_calc(self, year_count, day, dispatch_interval):
        '''
        Calculates the internal resistance of the cells based on an empirical model.

        Parameters
        ----------
        year_count : integer
            Current year iteration of the simulation.
        day : integer
            Current day in the year.
        dispatch_interval : integer
            Current dispatch interval in the day.

        Returns
        -------
        None

        '''
        
        # Calculate the number of months that have been simulated
        ST = (365*(year_count)+day+dispatch_interval/288)/30
        
        # Calculate the average SOC during the simulation
        if year_count+day+dispatch_interval == 0:
            SOC_avg = 0.5
        else:
            SOC_avg = (self.SOC_sum) / (self.dispatch_intervals) # dispatch_intervals is an attribute belonging to the general_systems class. It is part of the instance due to the storage_system constructor class 
            
        # Calculate the internal resistance
        self.R_cell = self.R_cell_initial + self.R_cell_initial*(6.9656*(10**(-8))*np.exp(0.05022*self.temp)) * (2.897*np.exp(0.006614*SOC_avg*100))*(ST**0.8)/100

    def eff_volt(self):
        '''
        Calculate the voltage efficiency of the battery.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        
        # Calculate the radicand
        sqrt_var = 0.25 - (abs(self.P_current)/(self.efficiency_sys*self.SOC_current*self.energy_capacity))*((self.U_batt_nom/self.U_batt)**2)*self.R_cell*(self.cell_e*self.U_cell_nom)/self.U_cell_nom
        
        # Error handling for negative radicand
        if sqrt_var < 0:
            print(sqrt_var)
            self.efficiency_volt = 0.5
        
        # Voltage efficiency based on positive radicand
        else:
            self.efficiency_volt = 0.5 + (sqrt_var)**0.5

    def SOC_max_aged(self, delT): 
        '''
        Update the state of health of the battery based on cycle or calendar ageing.
        Assumes all cells age at the same rate.

        Parameters
        ----------
        delT : float
            Length of the dispatch interval [h].

        Returns
        -------
        None

        '''
        # Define gas constant
        R = 8.314 #[J/mol/K]

        # Update filtered current value
        self.I_filt = 10**6*self.P_current / (74*6*8*33*self.U_batt)
        
        # Calculate cycle ageing based on empirical model and linear interpolation
        if self.I_filt < self.I_cyc:
            self.cycLossCurrentSum += self.I_filt
            self.cycLossIntervals += 1
            I_avg = abs(self.cycLossCurrentSum/self.cycLossIntervals)
            self.Ah_throughput += (self.SOC_current - self.SOC_pre)*self.cell_e
            
            if I_avg <= 1:
                B_cyc = 3.16*10**3
                
            elif I_avg <= 4:
                B_cyc = 3.16*10**3 + (I_avg - 1)*(2.17*10**4-3.16*10**3)/(3) #Ah^(1-z_cyc)
                
            elif I_avg <= 12:
                B_cyc = 2.17*10**4 + (I_avg - 4)*(1.29*10**4-2.17*10**4)/(8)
            
            else:
                B_cyc = 1.29*10**4 + (I_avg - 12)*(1.55*10**4-1.29*10**4)/(8)
                
            z_cyc = 0.55
            Ea_cyc = 31700 #J/mol
            alpha = 370.3 #J/mol/A
            
            C_cyc_loss = B_cyc*np.exp((-Ea_cyc + alpha*I_avg)/(R*self.temp))*((self.Ah_throughput)**z_cyc) # % per cell
            self.SOC_max_loss_cyc = C_cyc_loss/100
        
        # Calculate calendar ageing based on empirical model and linear interpolation
        else:
            self.calLossTime += delT
            self.calLossIntervals += 1
            self.calLossCurrentSum += self.SOC_current
            SOC_avg = self.calLossCurrentSum / self.calLossIntervals
            if SOC_avg <= 0.3:
                B_cal = 7.34*10**5 # Ah/s^z_cal
                z_cal = 0.943 
                Ea_cal = 73369 #J/mol
                
            elif SOC_avg <= 0.65:
                B_cal = 7.34*10**5 + (SOC_avg-0.3)*(6.75*10**5-7.34*10**5)/(0.35) # Ah/s^z_cal
                z_cal = 0.943 + (SOC_avg-0.3)*(0.9-0.943)/0.35 
                Ea_cal = 73369 + (SOC_avg-0.3)*(69804-73369)/(0.35) #J/mol
                
            else:
                B_cal = 6.75*10**5 + (SOC_avg-0.65)*(2.18*10**5 - 6.75*10**5)/(0.35) # Ah/s^z_cal
                z_cal = 0.9 + (SOC_avg-0.65)*(0.683 - 0.9)/(0.35)
                Ea_cal = 69804 + (SOC_avg-0.65)*(56937 - 69804)/(0.35) #J/mol
            
            C_cal_loss = B_cal*np.exp(-Ea_cal/(R*self.temp))*((self.calLossTime*3600)**z_cal) # % per cell
            self.SOC_max_loss_cal = C_cal_loss/100
        
        # Update the state of health of the system
        self.SOC_max = 1 - self.SOC_max_loss_cal - self.SOC_max_loss_cyc