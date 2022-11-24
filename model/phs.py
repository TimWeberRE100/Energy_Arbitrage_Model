import numpy as np
import constants as const

class pump_turb:
    def __init__(self, unit_type, phs_assumptions, index):
        if unit_type == "pump":
            index_pref = "g"
        elif unit_type == "turbine":
            index_pref = "h"
        else:
            print("Unknown pumped hydro component type")
        
        # Constant parameters initialised
        self.type = unit_type # Whether the object is a pump or a turbine
        self.id = index_pref + str(index)
        self.P_rated = phs_assumptions[id]["P_rated [MW]"] # Rated power of the pump/turbine [MW]
        self.Q_peak = phs_assumptions[id]["Q_peak [m3/s]"] # Flow rate of the pump/turbine at peak efficiency [m^3/s]
        self.efficiency_peak = phs_assumptions[id]["Efficiency_peak"] # Peak efficiency of the pump/turbine (0,1]

        # Current state parameters initialised
        self.Q_previous = 0 # Flow rate from the previous interval [m^3/s]
        self.P_previous = 0 # Power from the previous interval [MW]
        self.Q_t = 0 # Flow rate of the current interval [m^3/s]
        self.P_t = 0 # Power of the current interval [MW]

class phs:
    def __init__(self,assumptions, phs_assumptions):
        # Reservoir parameters initialised        
        self.V_res_u = int(assumptions["V_res_upper"]) # Total volume of the upper reservoir [m^3]
        self.V_res_l = int(assumptions["V_res_lower"]) # Total volume of the lower reservoir [m^3]
        self.H_r = int(assumptions["H_r"]) #
        self.H_ur = int(assumptions["H_ur"]) #
        self.H_lr = int(assumptions["H_lr"]) #

        # Pump parameters initialised
        self.g_range = int(assumptions["g_index_range"]) # Number of pumps attached to the penstock
        self.H_p_effective = int(assumptions["H_p_effective"]) # Effective head of pumps used during scheduling [m]
        self.H_pl_effective = int(assumptions["H_pl_effective"]) # Effective head loss of pumps assumed during scheduling [m]
        self.pump_penstock_d = float(assumptions["pump_penstock_pipe_diameter"]) # Diameter of the pump penstock [m]
        self.pump_penstock_l = float(assumptions["pump_penstock_pipe_length"]) # Length of the pump penstock [m]
        self.pump_K_fittings = float(assumptions["pump_K_fittings"]) #
        self.pump_abs_roughness = float(assumptions["pump_abs_roughness"])*10**(-3) #

        # Turbine parameters initialised
        self.h_range = int(assumptions["h_index_range"]) # Number of turbines attached to the penstock
        self.H_t_effective = int(assumptions["H_t_effective"]) # Effective head of turbines used during scheduling [m]
        self.H_tl_effective = int(assumptions["H_tl_effective"]) # Effective head loss of turbines assumed during scheduling
        self.turbine_penstock_d = float(assumptions["turbine_penstock_pipe_diameter"]) # Diameter of the turbine penstock [m]
        self.turbine_penstock_l = float(assumptions["turbine_penstock_pipe_length"]) # Length of the turbine penstock [m]
        self.turbine_K_fittings = float(assumptions["turbine_K_fittings"]) #
        self.turbine_abs_roughness = float(assumptions["turbine_abs_roughness"])*10**(-3) #

        # Assumed operating parameters initialised
        self.mol_lr = int(assumptions["MOL_lr"]) # Minimum operating level of lower reservoir [m]
        self.mol_ur = int(assumptions["MOL_ur"]) # Minimum operating level of upper reservoir [m]
        self.fsl_lr = self.H_lr # Full supply level of lower reservoir [m]
        self.fsl_ur = self.H_r + self.H_ur + self.H_lr # Full supply level of upper reservoir [m]
        self.rt_t_tnl = int(assumptions["RT_T_TNL"]) # Ramp time to transition from discharging to turbine no load [s]
        self.rt_tnl_p = int(assumptions["RT_TNL_P"]) # Ramp time to transition from turbine no load to pumping [s]
        self.rt_p_tnl = int(assumptions["RT_P_TNL"]) # Ramp time to transition from pumping to turbine no load [s]
        self.rt_tnl_t =int(assumptions["RT_TNL_T"]) # Ramp time to transition from turbine no load to discharging [s]

        # Current state parameters initialised
        self.pumps = [pump_turb("pump", phs_assumptions, g) for g in range(1,self.g_range+1)] # List of pump objects
        self.turbines = [pump_turb("turbine", phs_assumptions, h) for h in range(1,self.h_range+1)] # List of turbine objects
        self.Q_pump_penstock_t = 0 # Current flow rate in the pump penstock [m^3/s]
        self.Q_turbine_penstock_t = 0 # Current flow rate in the turbine penstock [m^3/s]
        self.H_pl_t = 0 # Current pump head loss [m]
        self.H_tl_t = 0 # Current turbine head loss [m]
        self.H_p_t = 0 # Current pump head [m]
        self.H_t_t = 0 # Current turbine head [m]

    ############### NEED TO REBUILD THE BELOW FUNCTIONS TO BE INSTANCE METHODS ############################
    def pumpHeadLoss(self):
        '''
        Calculate the pump head loss [m] for the PHS.

        Parameters
        ----------
        None.

        Returns
        -------
        self.H_pl_t : float
            Pump head loss based on pump flow rate for current dispatch interval.

        '''
        
        # Initialize the parameters
        g = 9.81 # m/s^2
        rho = 997 # kg/m^3
        mu = 8.9*10**(-4) # Pa*s
        
        # Define variables and calculate head loss
        v_p = self.Q_pump_penstock_t/(0.25*np.pi*(self.pump_penstock_d**2))
        Re_p = (rho*v_p*self.pump_penstock_d)/mu
        F_p = (1.8*np.log10(6.9/Re_p + ((self.pump_abs_roughness/self.pump_penstock_d)/3.7)**1.11))**(-2)
        K_pipe = (F_p*self.pump_penstock_l)/self.pump_penstock_d
        K_p = K_pipe + self.pump_K_fittings
        self.H_pl_t = K_p*(v_p**2)/(2*g) 

        return self.H_pl_t
        
    def turbineHeadLoss(self):
        '''
        Calculate the turbine head loss [m] for the PHS.

        Parameters
        ----------
        None.

        Returns
        -------
        self.H_tl_t : float
            Turbine head loss based on turbine flow rate for current dispatch interval.

        '''
        
        # Initialize the parameters
        g = 9.81 # m/s^2
        rho = 997 # kg/m^3
        mu = 8.9*10**(-4) # Pa*s
        
        # Define variables and calculate head loss
        if self.Q_turbine_penstock_t != 0:
            v_t = self.Q_turbine_penstock_t/(0.25*np.pi*(self.turbine_penstock_d**2))
            Re_t = (rho*v_t*self.turbine_penstock_d)/mu
            F_t = (1.8*np.log10(6.9/Re_t + ((self.turbine_abs_roughness/self.turbine_penstock_d)/3.7)**1.11))**(-2)
            K_pipe = (F_t*self.turbine_penstock_l)/self.turbine_penstock_d
            K_t = K_pipe + self.turbine_K_fittings
            self.H_tl_t = K_t*(v_t**2)/(2*g)
            
        else:
            self.H_tl_t = 0

        return self.H_tl_t

    def pumpHead(self):
        '''
        Calculate the net head for the pumps.

        Parameters
        ----------
        None.

        Returns
        -------
        self.H_p_t : float
            Net head for pump in dispatch interval

        '''
        
        # Calculate pump head
        self.H_p_t = (self.SOC_current*(self.fsl_ur - self.mol_ur) + self.mol_ur - ((1 - self.SOC_current)*(self.V_res_u/self.V_res_l)*(self.fsl_lr - self.mol_lr)) - self.mol_lr) + self.H_pl_t

        return self.H_p_t

    def Q_pump(self, pump_index):
        '''
        Flow rate for a pump unit.

        Parameters
        ----------
        pump_index : integer
            Index for the pump unit to which the flow rate corresponds.

        Returns
        -------
        self.pumps[pump_index].Q_t : float
            Pump flow rate during dispatch interval.

        '''

        self.pumpHead()
        
        if self.H_p_t < 0:
            self.pumps[pump_index].Q_t = 0
            
        else:    
            self.pumps[pump_index].Q_t = (self.pumps[pump_index].P_previous*self.pumps[pump_index].efficiency_peak*10**6)/(self.H_p_t*const.rho*const.gravity)
        
        self.Q_pump_penstock_t = sum([self.pumps[g].Q_t for g in range(1,self.g_range+1)])

        return self.pumps[pump_index].Q_t

    def Q_turbine(dischargingPower,eff_turbine,H_turbine,rho,gravity):
        '''
        Flow rate for a turbine unit

        Parameters
        ----------
        dischargingPower : float
            Dispatch power of the turbine.
        eff_turbine : float
            Turbine efficiency.
        H_turbine : float
            Net head of turbine.
        rho : integer
            Density of water [kg/m^3].
        gravity : float
            Acceleration due to gravity [m/s^2].

        Returns
        -------
        Q_turbine_t : float
            Turbine flow rate during interval.

        '''
        
        if (eff_turbine == 0) or (H_turbine < 0):
            Q_turbine_t = 0
        
        else:
            Q_turbine_t = (dischargingPower*10**6)/(H_turbine*rho*gravity*eff_turbine)
            
        return Q_turbine_t

    def turbineHead(SOC,system_assumptions,H_tl):
        '''
        Calculate net head for the turbines.

        Parameters
        ----------
        SOC : float
            State of charge of the system.
        system_assumptions : dictionary
            Dictionary of the assumed system parameters for the simulation.
        H_tl : float
            Turbine head loss [m].

        Returns
        -------
        turbineHead : float
            Net head of the turbines [m].

        '''
        # Define parameters
        H_r = float(system_assumptions["H_r"])
        H_ur = float(system_assumptions["H_ur"])
        H_lr = float(system_assumptions["H_lr"])
        volume_ur = int(system_assumptions["V_res_upper"])
        volume_lr = int(system_assumptions["V_res_lower"])
        
        FSL_ur = H_r + H_ur + H_lr
        MOL_ur = float(system_assumptions["MOL_ur"])
        FSL_lr = H_lr
        MOL_lr = float(system_assumptions["MOL_lr"])
        
        # Calculate turbine head
        turbineHead = (SOC*(FSL_ur - MOL_ur) + MOL_ur - ((1 - SOC)*(volume_ur/volume_lr)*(FSL_lr - MOL_lr)) - MOL_lr) - H_tl
        return turbineHead

    def pumpEfficiency(Q_pump_g):
        '''
        Calculate the pump efficiency.

        Parameters
        ----------
        Q_pump_g : float
            Pump flow rate [m^3/s].

        Returns
        -------
        pump_efficiency : float
            Efficiency of the pump unit.

        '''
        pump_efficiency = 0.91
        return pump_efficiency

    def turbineEfficiency(phs_assumptions,Q_turb_h,turbine_index):
        '''
        Calculate the turbine efficiency based on empirically fitted model.

        Parameters
        ----------
        phs_assumptions : dictionary
            Dictionary of the PHS pump and turbine assumptions.
        Q_turb_h : float
            Flow rate of turbine unit [m^3/s].
        turbine_index : integer
            Index 'h' of the turbine.

        Returns
        -------
        turbine_efficiency : float
            Efficiency of the turbine unit.

        '''
        # Define parameters
        Qpeak_turb_h = phs_assumptions["h"+str(turbine_index)]["Q_peak [m3/s]"]
        Q_ratio_turb_h = Q_turb_h / Qpeak_turb_h
        
        a = [-0.1421,3.044,-3.6718,2.3929,-0.7062]
        
        # Calculate turbine efficiency
        if Q_turb_h > 0.01:
            turbine_efficiency = sum(a[n]*(Q_ratio_turb_h**n) for n in range(0,len(a)))
        else:
            turbine_efficiency = 0
        return turbine_efficiency