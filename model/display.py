'''
Display the visual plots and print the statistics for the simulation.

Functions
---------
plotOutputs2d

schedulingOutputs

plotOutputs3d

chargingOutputsDay

statistics

chargingOutputsAnnual

chargingOutputsLifetime
'''

import matplotlib.pyplot as plt
import matplotlib as mp
import scipy.stats as spst
import pyomo.environ as pyo
from volatility import volatility

# Turn display on or off
display_arg = True

# Define the test day
test_day = 364

# Increase font size on plots
mp.rcParams.update({'font.size': 28,
                    'axes.labelpad': 12,
                    'figure.figsize': (16,10)})

def plotOutputs2d(x_values, y_values, x_title, y_title, legend_list):
    '''
    Generate a plot on 2 axes.

    Parameters
    ----------
    x_values : list
        List of lists that contain the x values for the data.
    y_values : list
        List of lists that contain the y values of the data.
    x_title : string
        Title of the x axis.
    y_title : string
        Title of the left-hand y axis.
    legend_list : list
        List of strings that contain the name of each data to be plotted on the figure.

    Returns
    -------
    None.

    Side-effects
    ------------
    Show the figure in a separate window.
    '''

    fig, ax1 = plt.subplots()
    
    y_axis_count = 1
    price_count = 0
    misc_count = 0
    power_count = 0
    
    legend1 = []
    legend2 = []

    x_limits = [0,0]
    y1_limits = [0,0]
    y2_limits = [0,0]

    ax1.set_xlabel(x_title)
    ax1.set_ylabel(y_title, color = 'tab:red')

    # Define the arguments for the plots
    for legend_index in range(0,len(legend_list)):
        axis_assignment = 1
        draw_style = 'steps-pre'

        # Define the colours of the curves
        if 'SOC' in legend_list[legend_index]:
            colour = 'tab:red'
            legend1.append(legend_list[legend_index])
        elif 'Price' in legend_list[legend_index]:
            colour_list = ['b', 'cornflowerblue']
            colour = colour_list[price_count]          
            price_count += 1
            axis_assignment = 2
            y_axis_count = 2
            legend2.append(legend_list[legend_index])
        elif 'Power' in legend_list[legend_index] or 'Bids' in legend_list[legend_index] or 'Offers' in legend_list[legend_index]:
            colour_list = ['tab:orange','tab:green','tab:purple','tab:pink','tab:olive','tab:cyan']
            colour = colour_list[power_count]
            power_count += 1
            legend1.append(legend_list[legend_index])            
        else:
            colour_list = ['rx','bx']
            colour = colour_list[misc_count]
            misc_count += 1
            draw_style = 'points'
            legend1.append(legend_list[legend_index])
        
        # If there is price data, create the second y-axis on the first iteration
        if (y_axis_count == 2) & (price_count == 1):
            ax2 = ax1.twinx()            
            ax2.set_ylabel("Price [$/MWh]", color = 'tab:blue')
            
        # Add the curve to the subplot
        if draw_style == 'points':
            ax1.plot(x_values[legend_index],y_values[legend_index],colour,markersize=20)
        else:
            if axis_assignment == 1:  
                x_limits[0] = min([x_limits[0], min(x_values[legend_index])])
                x_limits[1] = max([x_limits[1], max(x_values[legend_index])])
                y1_limits[0] = min([y1_limits[0], min(y_values[legend_index])])
                y1_limits[1] = max([y1_limits[1], max(y_values[legend_index])])
                x_adjust = (x_limits[1] - x_limits[0])*0.05
                y1_adjust = (y1_limits[1] - y1_limits[0])*0.05

                tr = mp.transforms.offset_copy(ax1.transData, fig=fig, x=2*legend_index, y=2*legend_index, units='points')
                ax1.plot(x_values[legend_index],y_values[legend_index], color = colour, drawstyle=draw_style, transform=tr)
                ax1.tick_params(axis='y', labelcolor=colour)
                ax1.set_xlim(left=x_limits[0] - x_adjust,right=x_limits[1] + x_adjust)
                ax1.set_ylim(bottom=y1_limits[0] - y1_adjust,top=y1_limits[1] + y1_adjust)
            
            elif axis_assignment == 2:
                x_limits[0] = min([x_limits[0], min(x_values[legend_index])])
                x_limits[1] = max([x_limits[1], max(x_values[legend_index])])
                y2_limits[0] = min([y1_limits[0], min(y_values[legend_index])])
                y2_limits[1] = max([y1_limits[1], max(y_values[legend_index])])
                x_adjust = (x_limits[1] - x_limits[0])*0.05
                y2_adjust = (y2_limits[1] - y2_limits[0])*0.05

                tr = mp.transforms.offset_copy(ax2.transData, fig=fig, x=2*legend_index, y=2*legend_index, units='points')
                ax2.plot(x_values[legend_index],y_values[legend_index],color = colour, drawstyle=draw_style, transform=tr)
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                ax1.set_xlim(left=x_limits[0] - x_adjust,right=x_limits[1] + x_adjust)
                ax2.set_ylim(bottom=y2_limits[0] - y2_adjust,top=y2_limits[1] + y2_adjust)
    
    # Add the legends to the plot if required
    if y_axis_count == 2:
        ax2.legend(legend2,loc='upper left', bbox_to_anchor=(0.1, 1))
    ax1.legend(legend1, loc='upper left', bbox_to_anchor=(0.1, 0.8))

    # Display the plots
    fig.tight_layout()
    plt.show()

def schedulingOutputs(solution_inst, storage_system_inst, dispatch_offers, dispatch_bids):
    '''
    Display the outputs for the scheduling algorithm on the test day.

    Parameters
    ----------
    solution_inst : 
        Object containing solution from the scheduling algorithm found using the CBC solver and Pyomo package.
    storage_system_inst : storage_system
        Object containing storage system attributes and current state.
    dispatch_offers : list
        List of dispatch offer prices made by the storage system for the trading day.
    dispatch_bids : list
        List of dispatch offer bids made by the storage system for the trading day.

    Returns
    -------
    None.

    Side-effects
    ------------
    Print the results from the scheduling algorithm on the chosen test day.
    Plot the scheduled SOC for the test day.
    Plot the scheduled capacities for the test day.
    Plot the bid and offer prices for the test day.
    '''

    # Define the output variables
    dispatch_offer_caps = []    # Capacity of offer at each trading interval
    dispatch_bid_caps = []      # Capacity of bid at each trading interval
    total_dispatch_caps = []    # Total scheduled capacity at each trading interval
    SPs = []                    # Pre-dispatch spot price at each trading interval
    SOCs = []                   # Scheduled SOC at each trading interval
    trading_intervals = []      # Trading interval count
    charging_behaviour = []     # Scheduled charging behavior at each trading interval (binary)
    turbine_flows = []          # Turbine penstock flow rate at each trading interval        
    turbine_losses = []         # Turbine penstock flow rate losses at each trading interval
    unit_g_capacities = {}      # Scheduled capacity of individual pumps at each trading interval
    unit_h_capacities = {}      # Scheduled capacity of individual turbines at each trading interval
    discharged_energy = []      # Scheduled discharged energy at each trading interval
    charged_energy = []         # Scheduled charged energy at each trading interval

    if storage_system_inst.type == "PHS":
        for g in solution_inst.g:
            unit_g_capacities[str(g)] = []
            
        for h in solution_inst.h:
            unit_h_capacities[str(h)] = []
    elif storage_system_inst.type == "BESS":
        pass

    for d in solution_inst.w:
        if storage_system_inst.type == "PHS":
            unit_h_subOffers = []   # Turbine offer capacities for trading interval d
            unit_h_subFlows = []    # Turbine flow rates for trading interval d
            unit_h_losses = []      # Turbine flow rate losses for trading interval d
            unit_g_subBids = []     # Pump flow rate capacities for trading interval d

            # Optimisation outputs
            for h in solution_inst.h:
                unit_h_subOffers.append(solution_inst.D[d,h].value)  
                unit_h_subFlows.append(solution_inst.Qt[d,h].value)
                unit_h_losses.append(solution_inst.QtLoss[d,h].value)
                unit_h_capacities[str(h)].append(solution_inst.D[d,h].value)
                
            for g in solution_inst.g:
                if solution_inst.C[d,g].value > 0.1:
                    unit_g_subBids.append(-storage_system_inst.pumps[g-1].P_rated)
                    unit_g_capacities[str(g)].append(-storage_system_inst.pumps[g-1].P_rated)
                else:
                    unit_g_subBids.append(0)
                    unit_g_capacities[str(g)].append(0)
                
            turbine_flows.append(unit_h_subFlows)
            turbine_losses.append(unit_h_losses)
            dispatch_offer_caps.append(unit_h_subOffers)
            dispatch_bid_caps.append(unit_g_subBids)
            
            # Define dispatch bid/offer for correlation 
            if solution_inst.D_tot[d].value > 0:
                total_dispatch_caps.append(solution_inst.D_tot[d].value)
            else:
                total_dispatch_caps.append(-solution_inst.C_tot[d].value)
            
        else:
            dispatch_offer_caps.append(solution_inst.D[d].value)
            dispatch_bid_caps.append(-solution_inst.C[d].value)
            
            # Define dispatch bid/offer for correlation 
            if solution_inst.D[d].value > 0:
                total_dispatch_caps.append(solution_inst.D[d].value)
            else:
                total_dispatch_caps.append(-solution_inst.C[d].value)
            
        charging_behaviour.append(solution_inst.w[d].value)
        discharged_energy.append(solution_inst.Ed[d].value)
        charged_energy.append(solution_inst.Ec[d].value)
        
        # Test variables
        SPs.append(float(solution_inst.SP[d][d]))
        trading_intervals.append(d)
        SOCs.append(solution_inst.SOC[d].value)
    
    print("----------- Test Day %d Scheduling Results -----------" % test_day)
    print("SOC: ",SOCs,"\n")
    print("Offer Capacities: ",dispatch_offer_caps,"\n")
    print("Bid Capacities: ",dispatch_bid_caps,"\n")
    print("Spot Price: ",SPs,"\n")
    print("Charging Behaviour: ",charging_behaviour,"\n")
    print("Arbitrage Revenue: ",pyo.value(solution_inst.arbitrageValue),"\n")
    print("Internal Energy Discharged: ", discharged_energy,"\n")
    print("Internal Energy Charged: ", charged_energy,"\n")

    if storage_system_inst.type == "PHS":
        print("Turbine Flow Rate: ",turbine_flows,"\n")
        print("Turbine Flow Rate Losses: ",turbine_losses,"\n")
        print("Pump Unit Capacities", unit_g_capacities,"\n")
        print("Turbine Unit Capacities", unit_h_capacities,"\n")
        
    # Plot scheduled SOC change
    legend_list = ["SOC", "Pre-dispatch Spot Price"]
    plotOutputs2d([trading_intervals,trading_intervals], [SOCs,SPs], "Trading Interval", "Scheduled State-of-charge", legend_list)

    # Plot scheduled capacities of units
    if storage_system_inst.type == "PHS":
        legend_list = ["Pump " + str(g) + " Power" for g in range(1,len(unit_g_capacities)+1)] + ["Turbine " + str(h) + " Power" for h in range(1,len(unit_h_capacities)+1)]
        legend_list.append("Pre-dispatch Spot Price")
        x_values = len(legend_list) * [trading_intervals]
        y_values = [unit_g_capacities[str(g)] for g in range(1, len(unit_g_capacities)+1)] + [unit_h_capacities[str(h)] for h in range(1,len(unit_h_capacities)+1)] + [SPs]
        plotOutputs2d(x_values, y_values, "Trading Interval", "Unit Scheduled Power [MW]", legend_list)
    else:
        legend_list = ["Charging Power", "Discharging Power", "Pre-dispatch Spot Price"]
        x_values = len(legend_list) * [trading_intervals]
        y_values = [dispatch_bid_caps] + [dispatch_offer_caps]
        plotOutputs2d(x_values, y_values, "Trading Interval", "Scheduled Power [MW]", legend_list)

    # Plot bid and offer prices
    legend_list = ["Bids", "Offers", "Pre-dispatch Spot Price"]
    x_values = len(legend_list) * [trading_intervals]
    y_values = [dispatch_bids] + [dispatch_offers] + [SPs]
    plotOutputs2d(x_values, y_values, "Trading Interval", "Bid/Offer Amount [$/MWh]", legend_list)

def plotOutputs3d(x_values, y_values, z_values, x_title, y_title, z_title, zlim_bool):
    '''
    Generate a plot of data along 3 axes.

    Parameters
    ----------
    x_values : list
        List of floats for x values of the data.
    y_values : list
        List of floats for y values of the data.
    z_values : list
        List of floats for z values of the data.
    x_title : string
        Title of the x axis.
    y_title : string
        Title of the y axis.
    z_title : string
        Title of the z axis.
    zlim_bool : bool
        True value denotes a limit of [0,1] on the z-axis

    Returns
    -------
    None.

    Side-effects
    ------------
    Show the 3-dimensional plot in a separate window.
    '''
    
    mp.rcParams.update({'axes.labelpad': 26})
                            
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x_values,y_values,z_values,cmap='viridis',edgecolor='none')
    
    if zlim_bool:
        ax.set_zlim(0,1)
    
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_zlabel(z_title)
                            
    plt.show()

def chargingOutputsDay(storage_system_inst, daily_memory):
    '''
    Display the outputs from the charging module on the test day.

    Parameters
    ----------
    storage_system_inst : storage_system
        Object containing storage system attributes and current state.
    daily_memory : memory_daily
        Object containing the data for the charging actitivty on the trading day.

    Returns
    -------
    None.

    Side-effects
    ------------
    Print the results from the charging module on the chosen test day.
    Plot the actual SOC for the test day.
    Plot the turbine/pump head against the SOC for the test day.
    Plot the turbine/pump head loss against the penstock flow rate for the test day.
    '''

    dispatchIntervals = list(range(1,289))
    
    # Plot SOC for a particular day    
    legend_list = ["SOC", "Dispatch Price"]
    x_values = len(legend_list) * [dispatchIntervals]
    y_values = [daily_memory.SOC_day] + [daily_memory.dispatch_prices]            
    plotOutputs2d(x_values, y_values, "Dispatch Interval", "Actual SOC", legend_list)
                
    if storage_system_inst.type == "PHS":
        # Plot head vs SOC
        legend_list = ["Turbine Head"]
        plotOutputs2d([daily_memory.SOC_day], [daily_memory.headTurbine], "Actual SOC", "Head [m]",legend_list)

        legend_list = ["Pump Head"]
        plotOutputs2d([daily_memory.SOC_day], [daily_memory.headPump], "Actual SOC", "Head [m]",legend_list)
        
        # Plot head loss vs penstock flow rate
        legend_list = ["Turbine Head Loss"]
        plotOutputs2d([daily_memory.flowRateTurbine], [daily_memory.headLossTurbine], "Flow rate [m^3/s]", "Head Loss [m]", legend_list)

        legend_list = ["Pump Head Loss"]
        plotOutputs2d([daily_memory.flowRatePump], [daily_memory.headLossPump], "Flow rate [m^3/s]", "Head Loss [m]", legend_list)        

def statistics(storage_system_inst, memory, timeframe):
    '''
    Calculate and print statistics for the simulation.

    Parameters
    ----------
    storage_system_inst : storage_system
        Object containing storage system attributes and current state.
    memory : memory
        Object containing simulation data.
    timeframe : string
        Either "annual" or "lifetime" to define the timeframe of the memory.

    Returns
    -------
    None.

    Side-effects
    ------------
    Print the annual or lifetime statistics for the simulation.
    '''

    # Calculate correlation coefficients
    SP_dispatch_intervals = []
    for i in memory.SP:
        for j in range(0,6):
            SP_dispatch_intervals.append(i)

    r1_SP = spst.kendalltau(SP_dispatch_intervals,memory.dispatched_capacity)
    r2_DP = spst.kendalltau(memory.DP,memory.dispatched_capacity)

    # Calculate price volatility
    DP_vol = volatility(memory.DP)
    SP_vol = volatility(memory.SP)

    # Display results
    if timeframe == "annual":
        print("---------- Annual Statistics ----------")
    elif timeframe == "lifetime":
        print("---------- Lifetime Statistics ----------")

    print("Dispatch Price Volatility (S.D.): ", DP_vol)
    print("Spot Price Volatility (S.D.): ", SP_vol)

    print("Kendall Tau (Dispatched Capacity vs Spot Price): ",r1_SP)
    print("Kendall Tau (Dispatched Capacity vs Dispatch Price): ",r2_DP)

    if storage_system_inst.type == "BESS":
        print("Final State-of-health: ", storage_system_inst.SOC_max)
        print("Final Internal Resistance of Cells: ", storage_system_inst.R_cell)

def chargingOutputsAnnual(storage_system_inst, annual_memory):
    '''
    Generate outputs for the simulation up until the test day for the year.

    Parameters
    ----------
    storage_system_inst : storage_system
        Object containing storage system attributes and current state.
    annual_memory : memory
        Object containing the simulation year data.

    Returns
    -------
    None.

    Side-effects
    ------------
    Print the statistics up until the test day for the year.
    Plot the SOC of the system up until the test_day for the year.
    Plot the 3D voltage efficiency graph if the system is a BESS.
    '''

    # Print the annual statistics
    statistics(storage_system_inst, annual_memory, "annual")

    # Plot SOC vs time
    legend_list = ["SOC","Dispatch Price"]
    dispatch_intervals = list(range(1,len(annual_memory.SOC)) + 1)
    x_values = len(legend_list) * [dispatch_intervals]
    y_values = [annual_memory.SOC] + [annual_memory.dispatch_prices] 
    plotOutputs2d(x_values, y_values, "Dispatch Interval", "State-of-charge", legend_list)

    # Plot 3d efficiency
    if storage_system_inst.type == "BESS":
        plotOutputs3d(annual_memory.SOC, annual_memory.powerMagnitude, annual_memory.voltage_efficiency, "State-of-charge", "Power [MW]", "Voltage Efficiency", True)

def chargingOutputsLifetime(storage_system_inst, simulation_memory):
    '''Generate outputs for the simulation over the system lifetime.'''
    # Print the annual statistics
    statistics(storage_system_inst, simulation_memory, "lifetime")