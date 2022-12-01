import matplotlib.pyplot as plt
import matplotlib as mp
from mpl_toolkits import mplot3d
import scipy.stats as spst

# Define the test day
test_day = 10

# Increase font size on plots
mp.rcParams.update({'font.size': 28,
                    'axes.labelpad': 12,
                    'figure.figsize': (16,10)})

def schedulingOutputs(solution_inst, storage_system_inst):
    pass

def plotOutputs2d(x_values, y_values, x_title, y_title, legend_list):
    fig, ax1 = plt.subplots()
    
    y_axis_count = 1
    price_count = 0
    misc_count = 0
    power_count = 0
    
    legend1 = []
    legend2 = []

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
        elif 'Power' in legend_list[legend_index]:
            colour_list = ['tab:orange','tab:green','tab:purple','tab:pink','tab:olive','tab:cyan']
            colour = colour_list[power_count]
            power_count += 1
            legend1.append(legend_list[legend_index])            
        else:
            colour_list = ['rx','bx']
            colour = colour_list[misc_count]
            misc_count += 1
            draw_style = 'points'
        
        # If there is price data, create the second y-axis on the first iteration
        if (y_axis_count == 2) & (price_count == 1):
            ax2 = ax1.twinx()            
            ax2.set_ylabel(y_title, color = 'tab:blue')
            
        # Add the curve to the subplot
        if draw_style == 'points':
            ax1.plot(x_values[legend_index],y_values[legend_index],colour,markersize=20)
        else:
            if axis_assignment == 1:            
                tr = mp.transforms.offset_copy(ax1.transData, fig=fig, x=2*legend_index, y=2*legend_index, units='points')
                ax1.plot(x_values[legend_index],y_values[legend_index], color = colour, drawstyle=draw_style, transform=tr)
                ax1.tick_params(axis='y', labelcolor=colour)
            
            elif axis_assignment == 2:
                tr = mp.transforms.offset_copy(ax2.transData, fig=fig, x=2*legend_index, y=2*legend_index, units='points')
                ax2.plot(x_values[legend_index],y_values[legend_index],color = colour, drawstyle=draw_style, transform=tr)
                ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # Add the legends to the plot if required
    if y_axis_count == 2:
        ax2.legend(legend2,loc='upper left', bbox_to_anchor=(0.1, 1))
    ax1.legend(legend1, loc='upper left', bbox_to_anchor=(0.1, 0.8))

    # Display the plots
    fig.tight_layout()
    plt.show()

def plotOutputs3d(x_values, y_values, z_values, x_title, y_title, z_title):
    mp.rcParams.update({'axes.labelpad': 26})
                            
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x_values,y_values,z_values,cmap='viridis',edgecolor='none')
    #ax.set_zlim(0,1)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_zlabel(z_title)
                            
    plt.show()

def statistics():
    # Calculate correlation coefficients
    #spst.kendalltau(annual_SP_t,annual_dispatchCapacity)
    pass