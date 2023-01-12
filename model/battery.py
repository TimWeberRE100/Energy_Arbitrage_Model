'''
Define the class and functions relevant to batteries.

Classes
-------
battery

Functions
---------
U_OCV_calc
'''

import numpy as np
import constants as const

def U_OCV_calc(SOC):
        '''
        Calculate the open-circuit voltage of the cells at a particular SOC.

        Parameters
        ----------
        SOC : float
            State of charge of the system.

        Returns
        -------
        U_cell : float
            Open circuit voltage of the cell

        Side-effects
        ------------
        None
        '''

        n1 = 5.163*np.exp(-(((SOC-1.794)/1.665)**2))
        n2 = 0.3296*np.exp(-(((SOC-0.6405)/0.3274)**2))
        n3 = 1.59*np.exp(-(((SOC-0.06475)/0.4406)**2))
        n4 = 5.184*np.exp(-(((SOC-(-0.531))/0.3059)**2))
        U_cell = sum([n1,n2,n3,n4])
        return U_cell

class battery:
    '''
    Input to the storage_system constructor for battery objects.
    '''
    def __init__(self, assumptions):
        '''
        Initialise attributes of a battery object.

        Parameters
        ----------
        assumptions : dataframe
            Dataframe of assumptions defined by the user in the ASSUMPTIONS.csv file.
        '''
        self.obj_type = "bess"

        # Assumed parameters initialised
        self.efficiency_sys = float(assumptions["efficiency_sys"]) # System efficiency of the battery, including HVAC and other auxiliaries (0,1]
        self.P_standby = float(assumptions["P_standby"]) # Power consumption of idling system [MW]
        self.temp = int(assumptions["Temp"]) # Temperature of the battery [K]
        self.eol = float(assumptions["eol"]) # State of health at which point the Li-ion battery is assumed to reach end of life [0,1)
        self.R_cell_initial = float(assumptions["R_cell_initial"]) # Initial internal resistance of the cells at start of simulation [Ohms]
        self.series_cells = int(assumptions["series_cells"]) # Number of cells in series
        self.parallel_cells = int(assumptions["parallel_cells"]) # Number of cells in parallel
        self.cell_e = float(assumptions["cell_energy_capacity [Ah]"]) # Energy capacity of each cell [Ah]
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
        self.SOC_sum = 0.5 # Cumulative sum of state of charge over battery lifetime for the calculation of aging
        self.dispatch_intervals = 0 # Count of dispatch intervals over battery lifetime for the calculation of aging
        self.U_cell = self.U_cell_nom # Open-circuit voltage of the cell [V]
        self.R_cell = self.R_cell_initial # Internal resistance of the cell [Ohms]
        self.U_batt = self.U_batt_nom # Open-circuit voltage of the battery [V]
        self.efficiency_volt = 0 # Voltage efficiency of the battery (0,1]
        self.I_filt = 0 # Filtered current through the cells [A]

    def U_OCV_assign(self, SOC):
        '''
        Calculate the open-circuit voltage of the cells at a particular SOC. Updates the battery's attribute.

        Parameters
        ----------
        SOC : float
            State of charge of the system.

        Returns
        -------
        None.

        Side-effects
        ------------
        Update self.U_cell

        Update self.U_batt
        '''
        self.U_cell = U_OCV_calc(SOC)
        self.U_batt = self.U_cell * self.series_cells

    def R_cell_calc(self, year_count, day, dispatch_interval, SOC):
        '''
        Calculate the internal resistance of the cells based on an empirical model.

        Parameters
        ----------
        year_count : integer
            Current year iteration of the simulation.
        day : integer
            Current day in the year.
        dispatch_interval : integer
            Current dispatch interval in the day.
        SOC : float
            State of charge of the system.

        Returns
        -------
        self.R_cell : float
            Internal resistance of the cells.

        Side-effects
        ------------
        Update self.SOC_sum
        '''
        
        # Calculate the number of months that have been simulated
        ST = (365*(year_count)+day+dispatch_interval/288)/30

        # Update the SOC_sum in accordance with the newly assigned SOC
        self.SOC_sum += SOC
        
        # Calculate the average SOC during the simulation
        if year_count+day+dispatch_interval == 0:
            SOC_avg = 0.5
        else:
            SOC_avg = (self.SOC_sum) / (self.dispatch_intervals) 
            
        # Calculate the internal resistance
        self.R_cell = self.R_cell_initial + self.R_cell_initial*(6.9656*(10**(-8))*np.exp(0.05022*self.temp)) * (2.897*np.exp(0.006614*SOC_avg*100))*(ST**0.8)/100

        return self.R_cell

    def eff_volt(self, SOC, power, energy_capacity):
        '''
        Calculate the voltage efficiency of the battery.

        Parameters
        ----------
        SOC : float
            State of charge of the system.
        power : float
            Power output of the system.
        energy_capacity : float
            Energy capacity of the system.

        Returns
        -------
        self.efficiency_volt : float
            Voltage efficiency of the battery object.

        Side-effects
        ------------
        None.
        '''
        
        # Calculate the radicand
        sqrt_var = 0.25 - (abs(power)/(self.efficiency_sys*SOC*energy_capacity))*((self.U_batt_nom/self.U_batt)**2)*self.R_cell*(self.cell_e*self.U_cell_nom)/self.U_cell_nom
        
        # Error handling for negative radicand
        if sqrt_var < 0:
            self.efficiency_volt = 0.5
        
        # Voltage efficiency based on positive radicand
        else:
            self.efficiency_volt = 0.5 + (sqrt_var)**0.5
        
        return self.efficiency_volt

    def SOC_max_aged(self, delT, SOC, SOC_pre, power): 
        '''
        Update the state of health of the battery based on cycle or calendar ageing, assuming all cells age at the same rate.

        Parameters
        ----------
        delT : float
            Length of the dispatch interval [h].
        SOC : float
            State of charge of the system.
        SOC_pre : float
            State of charge of the system in the previous dispatch interval.
        power : float
            Power output of the system.

        Returns
        -------
        self.SOC_max : float
            State of health of the battery object.

        Side-effects
        ------------
        Update self.I_filt
        Call self.U_OCV_assign
        Update self.cycLossCurrentSum
        Update self.cycLossIntervals
        Update self.Ah_throughput
        Update self.SOC_max_loss_cyc
        Update self.calLossTime
        Update self.calLossIntervals
        Update self.calLossCurrentSum
        '''
        # Update filtered current value
        self.I_filt = 10**6*power / (74*6*8*33*self.U_batt)

        # Update the open-circuit voltages
        self.U_OCV_assign(SOC)
        
        # Calculate cycle ageing based on empirical model and linear interpolation
        if self.I_filt < self.I_cyc:
            self.cycLossCurrentSum += self.I_filt
            self.cycLossIntervals += 1
            I_avg = abs(self.cycLossCurrentSum/self.cycLossIntervals)
            self.Ah_throughput += (SOC - SOC_pre)*self.cell_e
            
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

            C_cyc_loss = B_cyc*np.exp((-Ea_cyc + alpha*I_avg)/(const.R*self.temp))*((self.Ah_throughput)**z_cyc) # % per cell
            
            self.SOC_max_loss_cyc = C_cyc_loss/100
        
        # Calculate calendar ageing based on empirical model and linear interpolation
        else:
            self.calLossTime += delT
            self.calLossIntervals += 1
            self.calLossCurrentSum += SOC
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
            
            C_cal_loss = B_cal*np.exp(-Ea_cal/(const.R*self.temp))*((self.calLossTime*3600)**z_cal) # % per cell
            self.SOC_max_loss_cal = C_cal_loss/100
        
        # Update the state of health of the system
        self.SOC_max = 1 - self.SOC_max_loss_cal - self.SOC_max_loss_cyc

        return self.SOC_max
    
    def incrementDispatchInterval(self):
        '''
        Update the current dispatch interval for the system's life.
        '''
        self.dispatch_intervals += 1

    def testToCurrent(self):
        pass

    def idleInterval(self):
        pass