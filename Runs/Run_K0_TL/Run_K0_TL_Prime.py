#!/usr/bin/env python
# coding: utf-8

# # Run_K0_TL_prime
# 
# **Author**: Chris Sheehan

#%% Block 1: Grid Set-up

# Initial imports
print('Importing initial libraries...')
import inspect
import os

# Directories
print('Setting directories...')
script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_directory = script_directory.replace('\\runs\\run_k0_tl', '')

# Print 
print('Setting variables...')

# ENTER VARIABLES ############################################################
##############################################################################

# Geomorphic parameters
m_sp = 0.5
n_sp = 1
K = 2.0987045879077573e-06    # Stream power erodibility. Calculated per Sheehan et al., 2026 as mean catchment Ksp, here assumed to be representitive of bedrock. Units depend on m_sp and n_sp.
D = (K * 1000)             # Linear hillslope diffusivity. Calculated per Sheehan et al., 2026. m^2 / yr. 

# Import and handle grid
DEM_path = project_directory + '\DEM\Seminole_32616_DEM_Processed.asc'      # DEM location
dxy = 50                                                                    # DEM resolution
no_data_value = -99999                                                      # DEM no data value
pour_point_node = 811                                                       # Watershed outlet node, identified manually. Ideal beacause (A). Running  FlowAccumulator on raw DEM identifies this node, and (B). It is a boundary node. Chestatee = 327. A = 3959.
# gauge_node = 439643                                                         # Real-world gauge location. Node identified manually
inflow_node = 1639165                                                       # Inflow node from upstream dam. Node identified manually.
inflow_drainage_area = 1.93213E10                                           # Total, undamed drainage area above dam. 

# 10Be data
Be10_Uplift = 9.15E-6                   # From Reusser et al., 2015, recalculated by Octopus. m / yr^-1
Be10_Uplift_min = 9.15E-6 - 2.15E-6     # From Reusser et al., 2015, recalculated by Octopus. m / yr^-1
Be10_Uplift_max = 9.15E-6 + 2.15E-6     # From Reusser et al., 2015, recalculated by Octopus. m / yr^-1

# Seg heads
seg1_head = 857436          # Node ids for the channel heads of interest. More or less correspond to segments analyzed in Sheehan et al., 2026
seg2_head = 1891905         # Node ids for the channel heads of interest. More or less correspond to segments analyzed in Sheehan et al., 2026
seg3_head = inflow_node     # Node ids for the channel heads of interest. More or less correspond to segments analyzed in Sheehan et al., 2026

# Plotting options
min_drainage_area = (dxy**2) * 1000     # Threshold drainage area to define channel extraction

# Export options
Directory = script_directory + '/'  # Export directory
alpha = 0.8                         # Drape transparency for hillshaded maps
Export_format = 'png'               # Export format
dpi = 150                           # Export dpi

##############################################################################
# ENTER VARIABLES ############################################################

# Import libraries
print('Importing primary libraries...')
import numpy as np
from numpy import nan, isnan
import pandas as pd
import math
import os
import os.path
from os import path
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
from landlab import RasterModelGrid, imshow_grid, imshowhs_grid
from landlab.plot.video_out import VideoPlotter
from landlab.components import PriorityFloodFlowRouter, SpaceLargeScaleEroder, SinkFillerBarnes, PerronNLDiffuse, StreamPowerEroder, Space, DepressionFinderAndRouter, LinearDiffuser, TaylorNonLinearDiffuser,  FlowAccumulator, ChannelProfiler, SteepnessFinder, ChiFinder, Lithology, LithoLayers, NormalFault
from landlab.io.esri_ascii import write_esri_ascii
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as colors
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from landlab.plot.graph import plot_graph
from landlab.io.esri_ascii import read_esri_ascii 
import EET
import matplotlib.cbook as cbook

# Set random seed
print('Setting random seed...')
np.random.seed(0) 

# Create directories
print('Creating export directories...')
if path.exists(str(Directory)+'/TerrainSandbox') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox')
if path.exists(str(Directory)+'/TerrainSandbox/CSV') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/CSV')

# Import DEM
print('Importing DEM...')
(mg, zr) = read_esri_ascii(DEM_path, name='topographic__elevation')

# Handle model DEM dimensions and non-value nodes
print('Handling DEM dimensions and non-value nodes...')
mg.set_nodata_nodes_to_closed(zr, no_data_value)
no_data_nodes = np.where(mg.at_node['topographic__elevation'] == no_data_value)
no_data_nodes = no_data_nodes[0]

# Create node keys
print('Creating soil and bedrock fields...')
mg.add_zeros('node', 'soil__depth')
mg.add_zeros('node', 'bedrock__depth')

# Handle Grid Boundaries
print('Handling grid boundaries...')
mg.set_status_at_node_on_edges(right=4, top=4, left=4, bottom=4)
mg.status_at_node[pour_point_node] = mg.BC_NODE_IS_FIXED_VALUE

# Fill Sinks
print('Filling sinks...')
sfb = SinkFillerBarnes(mg, method="D8", fill_flat=False)
sfb.run_one_step()

# Initialize FlowAccumulator
print('Initializing FlowAccumulator...')
frr = PriorityFloodFlowRouter(mg, flow_metric='D8')
frr.run_one_step()

# Set uplift rate
print('Setting uplift rate...')
U = np.ones(mg.number_of_nodes) * Be10_Uplift

# # Find nodes within radius of gauge
# radius = 200
# print('Finding nodes within ', radius, ' meters of gauge...')
# distance = np.copy(mg.at_node['topographic__elevation'])
# distance[:] = nan
# for i in np.arange(0, np.size(distance)):
#     r = ( ((mg.x_of_node[i] - mg.x_of_node[gauge_node]) ** 2) + ((mg.y_of_node[i] - mg.y_of_node[gauge_node]) ** 2) ) ** 0.5
#     distance[i] = r
# gauge_radius_nodes = np.where(distance < radius)
# gauge_radius_nodes = gauge_radius_nodes[0]

# Find node immediately upstream of pourpoint
radius = 200
print('Finding nodes within ', radius, ' meters of outlet...')
distance = np.copy(mg.at_node['topographic__elevation'])
distance[:] = nan
for i in np.arange(0, np.size(distance)):
    r = ( ((mg.x_of_node[i] - mg.x_of_node[pour_point_node]) ** 2) + ((mg.y_of_node[i] - mg.y_of_node[pour_point_node]) ** 2) ) ** 0.5
    distance[i] = r
outlet_radius_nodes = np.where(distance < radius)
outlet_radius_nodes = outlet_radius_nodes[0]

# Print space
print(' ')

#%% Block 2: Import DTB (Depth to Bedrock) and Set as soil__depth

# Import
print('Importing and handling DTB...')
DTB_path = project_directory + '\DTB\DTB_32616_50m_Clip.asc'
(mg2, dtb) = read_esri_ascii(DTB_path, name='dtb')
mg2.set_nodata_nodes_to_closed(dtb, no_data_value)
mg2.at_node['dtb'] /= 100                                       # Raw dataset is in cm. Convert to m.
mg.at_node['soil__depth'] = mg2.at_node['dtb']
mg.at_node['bedrock__elevation'] = mg.at_node['topographic__elevation'] - mg.at_node['soil__depth']

# Print space
print(' ')

#%% Block 3: Store initial conditions

# Initial conditions
print('Storing initial topographic__elevation, bedrock__elevation, and soil__depth...')
zr0 = np.copy(mg.at_node['topographic__elevation'])
br0 = np.copy(mg.at_node['bedrock__elevation'])
sd0 = np.copy(mg.at_node['soil__depth'])

# Print space
print(' ')

#%% Block 4: Set-up Channel Segments

# Print
print('Handling channel segments of interest...')

# Extract initial seg1
receivers = mg.at_node["flow__receiver_node"]
start_node = seg1_head
seg1_nodes = [start_node]
current_node = start_node
while True:
    next_node = receivers[current_node]
    if next_node == current_node:
        break  # reached the outlet
    seg1_nodes.append(next_node)
    current_node = next_node
seg1_nodes = np.flip(np.array(seg1_nodes))
#
seg1_distance = seg1_nodes * 0
for i in np.arange(1, len(seg1_nodes)):
    seg1_distance[i] = seg1_distance[i-1] + ( ((mg.x_of_node[seg1_nodes[i]] - mg.x_of_node[seg1_nodes[i-1]]) ** 2) + ((mg.y_of_node[seg1_nodes[i]] - mg.y_of_node[seg1_nodes[i-1]]) ** 2) ) ** 0.5

# Extract initial seg2
receivers = mg.at_node["flow__receiver_node"]
start_node = seg2_head
seg2_nodes = [start_node]
current_node = start_node
while True:
    next_node = receivers[current_node]
    if next_node == current_node:
        break  # reached the outlet
    seg2_nodes.append(next_node)
    current_node = next_node
seg2_nodes = np.flip(np.array(seg2_nodes))
#
seg2_distance = seg2_nodes * 0
for i in np.arange(1, len(seg2_nodes)):
    seg2_distance[i] = seg2_distance[i-1] + ( ((mg.x_of_node[seg2_nodes[i]] - mg.x_of_node[seg2_nodes[i-1]]) ** 2) + ((mg.y_of_node[seg2_nodes[i]] - mg.y_of_node[seg2_nodes[i-1]]) ** 2) ) ** 0.5

# Extract initial seg3
receivers = mg.at_node["flow__receiver_node"]
start_node = seg3_head
seg3_nodes = [start_node]
current_node = start_node
while True:
    next_node = receivers[current_node]
    if next_node == current_node:
        break  # reached the outlet
    seg3_nodes.append(next_node)
    current_node = next_node
seg3_nodes = np.flip(np.array(seg3_nodes))
#
seg3_distance = seg3_nodes * 0
for i in np.arange(1, len(seg3_nodes)):
    seg3_distance[i] = seg3_distance[i-1] + ( ((mg.x_of_node[seg3_nodes[i]] - mg.x_of_node[seg3_nodes[i-1]]) ** 2) + ((mg.y_of_node[seg3_nodes[i]] - mg.y_of_node[seg3_nodes[i-1]]) ** 2) ) ** 0.5

# Plot initial seg1
if path.exists(str(Directory)+'/TerrainSandbox/Seg1') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Seg1')
plt.ioff()
fig = plt.figure()        
plt.plot(seg1_distance / 1000, mg.at_node['topographic__elevation'][seg1_nodes], 'r--', linewidth = 0.5)
plt.plot(seg1_distance / 1000, mg.at_node['bedrock__elevation'][seg1_nodes], 'k--', linewidth = 0.5)
plt.xlim(0, 56)
plt.ylim(0, 80)
plt.grid()
plt.xlabel('Distance (km)')
plt.ylabel('Elevation (m)')
plt.title('t = 0 years')
plt.legend(['Topographic elevation', 'Bedrock elevation'], loc = 'upper left')
plt.tight_layout()
if path.exists(str(Directory)+'/TerrainSandbox/Seg1/Full') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Seg1/Full')
fig.savefig(str(Directory)+'/TerrainSandbox/Seg1/Full/0.'+Export_format,  format=Export_format, dpi=dpi)
#
if path.exists(str(Directory)+'/TerrainSandbox/Seg1/Zone_1') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Seg1/Zone_1')
plt.xlim(0, 15)
plt.ylim(10, 40)
fig.savefig(str(Directory)+'/TerrainSandbox/Seg1/Zone_1/0.'+Export_format,  format=Export_format, dpi=dpi)
#
plt.close(fig)

# Plot initial seg2
if path.exists(str(Directory)+'/TerrainSandbox/Seg2') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Seg2')
plt.ioff()
fig = plt.figure()        
plt.plot(seg2_distance / 1000, mg.at_node['topographic__elevation'][seg2_nodes], 'r--', linewidth = 0.5)
plt.plot(seg2_distance / 1000, mg.at_node['bedrock__elevation'][seg2_nodes], 'k--', linewidth = 0.5)
plt.xlim(0, 125)
plt.ylim(0, 130)
plt.grid()
plt.xlabel('Distance (km)')
plt.ylabel('Elevation (m)')
plt.title('t = 0 years')
plt.legend(['Topographic elevation', 'Bedrock elevation'], loc = 'upper left')
plt.tight_layout()
if path.exists(str(Directory)+'/TerrainSandbox/Seg2/Full') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Seg2/Full')
fig.savefig(str(Directory)+'/TerrainSandbox/Seg2/Full/0.'+Export_format,  format=Export_format, dpi=dpi)
#
if path.exists(str(Directory)+'/TerrainSandbox/Seg2/Zone_2') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Seg2/Zone_2')
plt.xlim(61, 99)
plt.ylim(10, 90)
fig.savefig(str(Directory)+'/TerrainSandbox/Seg2/Zone_2/0.'+Export_format,  format=Export_format, dpi=dpi)
#
plt.close(fig)

# Plot initial seg3
if path.exists(str(Directory)+'/TerrainSandbox/Seg3') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Seg3')
plt.ioff()
fig = plt.figure()        
plt.plot(seg3_distance / 1000, mg.at_node['topographic__elevation'][seg3_nodes], 'r--', linewidth = 0.5)
plt.plot(seg3_distance / 1000, mg.at_node['bedrock__elevation'][seg3_nodes], 'k--', linewidth = 0.5)
plt.xlim(0, 90)
plt.ylim(0, 45)
plt.grid()
plt.xlabel('Distance (km)')
plt.ylabel('Elevation (m)')
plt.title('t = 0 years')
plt.legend(['Topographic elevation', 'Bedrock elevation'], loc = 'upper left')
plt.tight_layout()
if path.exists(str(Directory)+'/TerrainSandbox/Seg3/Full') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Seg3/Full')
fig.savefig(str(Directory)+'/TerrainSandbox/Seg3/Full/0.'+Export_format,  format=Export_format, dpi=dpi)
#
if path.exists(str(Directory)+'/TerrainSandbox/Seg3/Zone_3') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Seg3/Zone_3')
plt.xlim(83, 88)
plt.ylim(15, 45)
fig.savefig(str(Directory)+'/TerrainSandbox/Seg3/Zone_3/0.'+Export_format,  format=Export_format, dpi=dpi)
#
plt.close(fig)

# Plot Segs_combined
if path.exists(str(Directory)+'/TerrainSandbox/Segs_combined') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Segs_combined')
#
plt.ioff()
fig = plt.figure() 
#
plt.plot(seg1_distance / 1000, zr0[seg1_nodes], '--', linewidth = 0.5, color = 'orange')     
plt.plot(seg1_distance / 1000, br0[seg1_nodes], 'k--', linewidth = 0.5)
#
plt.plot(seg2_distance / 1000, zr0[seg2_nodes], '--', linewidth = 0.5, color = 'orange')      
plt.plot(seg2_distance / 1000, br0[seg2_nodes], 'k--', linewidth = 0.5)
#
plt.plot(seg3_distance / 1000, zr0[seg3_nodes], '--', linewidth = 0.5, color = 'orange')     
plt.plot(seg3_distance / 1000, br0[seg3_nodes], 'k--', linewidth = 0.5)
#
plt.xlim(0, 125)
plt.ylim(0, 130)
#
plt.grid()
plt.xlabel('Distance (km)')
plt.ylabel('Elevation (m)')
plt.title('t = 0 years')
plt.legend(['Topographic elevation', 'Bedrock elevation'], loc = 'upper left')
plt.tight_layout()
#
# Plot full
if path.exists(str(Directory)+'/TerrainSandbox/Segs_combined/Full') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Segs_combined/Full')
fig.savefig(str(Directory)+'/TerrainSandbox/Segs_combined/Full/0.'+Export_format,  format=Export_format, dpi=dpi)
#
plt.close(fig)  

# Create seg_Maps
if path.exists(str(Directory)+'/TerrainSandbox/seg_Maps') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/seg_Maps')
plt.ioff()
fig = plt.figure()  
ax = imshowhs_grid(mg, "topographic__elevation", plot_type='Drape1', drape1=mg.at_node['topographic__elevation'],cmap='terrain',alpha=alpha,color_for_background='k',color_for_closed='k',allow_colorbar=True,cbar_loc="right",cbar_or='vertical',bbox_to_anchor=(1.05, 0.0, 0.03, 1.0),cbar_width="100%", cbar_height="100%", var_name='Elevation (m)', colorbar_label_x=1.5, colorbar_label_y=0.5, cbar_label_fontweight='regular', add_label_bbox=False)
formatter = ScalarFormatter()
formatter.set_powerlimits((-3, 3))  # Switch to scientific notation if the range is <10^-3 or >10^3
ax.xaxis.set_major_formatter(formatter)
ax.plot(mg.x_of_node[seg1_nodes], mg.y_of_node[seg1_nodes], 'r.', markersize = 1.5)
ax.plot(mg.x_of_node[seg2_nodes], mg.y_of_node[seg2_nodes], 'g.', markersize = 1.5)
ax.plot(mg.x_of_node[seg3_nodes], mg.y_of_node[seg3_nodes], 'b.', markersize = 1.5)
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/seg_Maps/Full.'+Export_format,  format=Export_format, dpi=dpi)
#
ax.set_xlim([683140, 695270])
ax.set_ylim([3427940, 3440170]) 
fig.savefig(str(Directory)+'/TerrainSandbox/seg_Maps/Zone_1.'+Export_format,  format=Export_format, dpi=dpi)
#
ax.set_xlim([664880, 682180])
ax.set_ylim([3475450, 3492060]) 
fig.savefig(str(Directory)+'/TerrainSandbox/seg_Maps/Zone_2.'+Export_format,  format=Export_format, dpi=dpi)
#
ax.set_xlim([681244, 686716])
ax.set_ylim([3495674, 3501314])
fig.savefig(str(Directory)+'/TerrainSandbox/seg_Maps/Zone_3.'+Export_format,  format=Export_format, dpi=dpi)  
#
plt.close(fig)

# Print space
print(' ')

#%% Block 5: Set-up Cross-sections

# Print 
print('Initializing cross-sections...')

# Cross-section A
xs_A0 = 110336          # 343566
xs_A1 = 152272          # 311193
#
x0 = mg.x_of_node[xs_A0]
y0 = mg.y_of_node[xs_A0]
x1 = mg.x_of_node[xs_A1]
y1 = mg.y_of_node[xs_A1]
#
dx = abs(x1 - x0)
dy = abs(y1 - y0)
sx = 1 if x0 < x1 else -1
sy = 1 if y0 < y1 else -1
err = dx - dy
#
xs_A_x = []
xs_A_y = []
while True:
    xs_A_x.append(x0)
    xs_A_y.append(y0)
    if x0 == x1 and y0 == y1:
        break
    e2 = 2 * err
    if e2 > -dy:
        err -= dy
        x0 += sx
    if e2 < dx:
        err += dx
        y0 += sy
#
xs_A_x = np.array(xs_A_x)
xs_A_y = np.array(xs_A_y)
#
xs_A_distance = xs_A_x * 0
for i in np.arange(1, len(xs_A_distance)):
    xs_A_distance[i] = xs_A_distance[i - 1] + ( ((xs_A_x[i] - xs_A_x[i - 1]) ** 2) + ((xs_A_y[i] - xs_A_y[i - 1]) ** 2) ) ** 0.5
#
xs_A_nodes = mg.find_nearest_node([xs_A_x, xs_A_y])

# Cross-section B
xs_B0 = 1228776                         # START HERE 9/9
xs_B1 = 1306748
#
x0 = mg.x_of_node[xs_B0]
y0 = mg.y_of_node[xs_B0]
x1 = mg.x_of_node[xs_B1]
y1 = mg.y_of_node[xs_B1]
#
dx = abs(x1 - x0)
dy = abs(y1 - y0)
sx = 1 if x0 < x1 else -1
sy = 1 if y0 < y1 else -1
err = dx - dy
#
xs_B_x = []
xs_B_y = []
while True:
    xs_B_x.append(x0)
    xs_B_y.append(y0)
    if x0 == x1 and y0 == y1:
        break
    e2 = 2 * err
    if e2 > -dy:
        err -= dy
        x0 += sx
    if e2 < dx:
        err += dx
        y0 += sy
#
xs_B_x = np.array(xs_B_x)
xs_B_y = np.array(xs_B_y)
#
xs_B_distance = xs_B_x * 0
for i in np.arange(1, len(xs_B_distance)):
    xs_B_distance[i] = xs_B_distance[i - 1] + ( ((xs_B_x[i] - xs_B_x[i - 1]) ** 2) + ((xs_B_y[i] - xs_B_y[i - 1]) ** 2) ) ** 0.5
#
xs_B_nodes = mg.find_nearest_node([xs_B_x, xs_B_y])

# Cross-section C
xs_C0 = 1586025          
xs_C1 = 1604151          
#
x0 = mg.x_of_node[xs_C0]
y0 = mg.y_of_node[xs_C0]
x1 = mg.x_of_node[xs_C1]
y1 = mg.y_of_node[xs_C1]
#
dx = abs(x1 - x0)
dy = abs(y1 - y0)
sx = 1 if x0 < x1 else -1
sy = 1 if y0 < y1 else -1
err = dx - dy
#
xs_C_x = []
xs_C_y = []
while True:
    xs_C_x.append(x0)
    xs_C_y.append(y0)
    if x0 == x1 and y0 == y1:
        break
    e2 = 2 * err
    if e2 > -dy:
        err -= dy
        x0 += sx
    if e2 < dx:
        err += dx
        y0 += sy
#
xs_C_x = np.array(xs_C_x)
xs_C_y = np.array(xs_C_y)
#
xs_C_distance = xs_C_x * 0
for i in np.arange(1, len(xs_C_distance)):
    xs_C_distance[i] = xs_C_distance[i - 1] + ( ((xs_C_x[i] - xs_C_x[i - 1]) ** 2) + ((xs_C_y[i] - xs_C_y[i - 1]) ** 2) ) ** 0.5
#
xs_C_nodes = mg.find_nearest_node([xs_C_x, xs_C_y])

# Plot initial xs_A
if path.exists(str(Directory)+'/TerrainSandbox/xs_A') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/xs_A')
plt.ioff()
fig = plt.figure()        
plt.plot(xs_A_distance, zr0[xs_A_nodes], 'r--', linewidth = 0.5)
plt.plot(xs_A_distance, br0[xs_A_nodes], 'k--', linewidth = 0.5)
plt.ylim(10, 50)
plt.grid()
plt.xlabel('Distance (m)')
plt.ylabel('Elevation (m)')
plt.title('t = 0 years')
plt.legend(['Topographic elevation', 'Bedrock elevation'], loc = 'upper left')
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/xs_A/0.'+Export_format,  format=Export_format, dpi=dpi)
plt.close(fig)

# Plot initial xs_B
if path.exists(str(Directory)+'/TerrainSandbox/xs_B') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/xs_B')
plt.ioff()
fig = plt.figure()        
plt.plot(xs_B_distance, zr0[xs_B_nodes], 'r--', linewidth = 0.5)
plt.plot(xs_B_distance, br0[xs_B_nodes], 'k--', linewidth = 0.5)
plt.ylim(35, 120)
plt.grid()
plt.xlabel('Distance (m)')
plt.ylabel('Elevation (m)')
plt.title('t = 0 years')
plt.legend(['Topographic elevation', 'Bedrock elevation'], loc = 'upper left')
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/xs_B/0.'+Export_format,  format=Export_format, dpi=dpi)
plt.close(fig)

# Plot initial xs_C
if path.exists(str(Directory)+'/TerrainSandbox/xs_C') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/xs_C')
plt.ioff()
fig = plt.figure()        
plt.plot(xs_C_distance, zr0[xs_C_nodes], 'r--', linewidth = 0.5)
plt.plot(xs_C_distance, br0[xs_C_nodes], 'k--', linewidth = 0.5)
plt.ylim(15, 80)
plt.grid()
plt.xlabel('Distance (m)')
plt.ylabel('Elevation (m)')
plt.title('t = 0 years')
plt.legend(['Topographic elevation', 'Bedrock elevation'], loc = 'upper left')
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/xs_C/0.'+Export_format,  format=Export_format, dpi=dpi)
plt.close(fig)

# Plot xs_Maps
if path.exists(str(Directory)+'/TerrainSandbox/xs_Maps') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/xs_Maps')
plt.ioff()
fig = plt.figure() 
ax = imshowhs_grid(mg, "topographic__elevation", plot_type='Drape1', drape1=mg.at_node['topographic__elevation'],cmap='terrain',alpha=alpha,color_for_background='k',color_for_closed='k',allow_colorbar=True,cbar_loc="right",cbar_or='vertical',bbox_to_anchor=(1.05, 0.0, 0.03, 1.0),cbar_width="100%", cbar_height="100%", var_name='Elevation (m)', colorbar_label_x=1.5, colorbar_label_y=0.5, cbar_label_fontweight='regular', add_label_bbox=False)
ax.plot(mg.x_of_node[xs_A_nodes], mg.y_of_node[xs_A_nodes], 'mo')
ax.plot(mg.x_of_node[xs_B_nodes], mg.y_of_node[xs_B_nodes], 'mo')
ax.plot(mg.x_of_node[xs_C_nodes], mg.y_of_node[xs_C_nodes], 'mo')
formatter = ScalarFormatter()
formatter.set_powerlimits((-3, 3))  # Switch to scientific notation if the range is <10^-3 or >10^3
ax.xaxis.set_major_formatter(formatter)
#
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/xs_Maps/Full.' + Export_format,  format=Export_format, dpi=dpi)
#
ax.set_xlim([683140, 695270])
ax.set_ylim([3427940, 3440170])  
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/xs_Maps/Zone_1.' + Export_format,  format=Export_format, dpi=dpi)  
#
ax.set_xlim([664880, 682180])
ax.set_ylim([3475450, 3492060])  
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/xs_Maps/Zone_2.' + Export_format,  format=Export_format, dpi=dpi) 
#
ax.set_xlim([681244, 686716])
ax.set_ylim([3495674, 3501314])  
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/xs_Maps/Zone_3.' + Export_format,  format=Export_format, dpi=dpi) 
#
plt.close(fig) 

# Print space
print(' ')


#%% Block 6: Reset Clocks

# Print 
print('Zeroing clocks...')

# Zero clocks
total_time = 0 
Plot_Ticker = 0
Export_DEM_Ticker = 0
timestep_integer = 0

# Print space
print(' ')

#%% Block 7: Time Parameters

# ENTER VARIABLES ############################################################
##############################################################################

dt = 100          
tmax = 1E6 

##############################################################################
# ENTER VARIABLES ############################################################

#OPERATORS-->DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE-----------------------------

# Print 
print('Setting time parameters...')

# Set t
t = np.arange(0, tmax, dt) 

# Print space
print(' ')

#%% Blovk 8: Plotting Options

# ENTER VARIABLES ############################################################
##############################################################################

# Intervals
Plot_interval = 2000
Export_DEM_Interval = 2000

# Toggle node fields
Plot__topographic__elevation = True
Plot__dbedelevdt = True
Plot__dsoildepthdt = True
Plot__dzdt = True
Plot__erosionrate = True
Plot__netdz = True
Plot__sediment__flux = True

# Toggle sediment flux timeseries
sediment__flux_timeseries_gauge = True
sediment__flux_timeseries_outlet = True

# Toggle segments
Plot__Seg1 = True
Plot__Seg2 = True
Plot__Seg3 = True
Plot__Segs_combined = True

# Toggle cross-sections
Plot__xs_A = True
Plot__xs_B = True
Plot__xs_C = True

# Toggle DEM export
Export_DEM = False

##############################################################################
# ENTER VARIABLES ############################################################

# Print
print('Setting plot parameters')

# Reset tickers
Plot_Ticker = 0
Export_DEM_Ticker = 0

# Enforce colorbar tickmark label scientific notation decimal places to 2 
def sci_notation_format(x, _):
    return f"{x:.2e}"

# Print space
print(' ')

#%% BLOCK 8: TIME LOOP

# Initialize linear diffuser
print('Initializing LinearDiffuser...') 
dfn = LinearDiffuser(mg, linear_diffusivity = D, method = 'simple', deposit = False)        

# Initialize SPACE
print('Initializing SLSE...') 
spr = SpaceLargeScaleEroder(mg, K_sed = K * 1.65, K_br = K, F_f = 0.0, phi = 0.3, H_star = 0.1, v_s = 1.0, v_s_lake = None, m_sp = m_sp, n_sp = n_sp, sp_crit_sed = 0.0, sp_crit_br = 0.0, discharge_field = "surface_water__discharge", erode_flooded_nodes = False, thickness_lim = 100)

# Initialize ErosionElevationTracker
print('Initializing ErosionElevationTracker...') 
eet = EET.ErosionElevationTracker(mg, bedrock__and__soil = True)

# Initialize appendable arrays
print('Initializing appendable arrays') 
timesteps = []
times = []
#
z_means = []
z_medians = []
z_2s = []
z_98s = []
z_mins = []
z_maxs = []
#
dbedelevdt_means = []
dbedelevdt_medians = []
dbedelevdt_2s = []
dbedelevdt_98s = []
dbedelevdt_mins = []
dbedelevdt_maxs = []
#
dsoildepthdt_means = []
dsoildepthdt_medians = []
dsoildepthdt_2s = []
dsoildepthdt_98s = []
dsoildepthdt_mins = []
dsoildepthdt_maxs = []
#
dzdt_means = []
dzdt_medians = []
dzdt_2s = []
dzdt_98s = []
dzdt_mins = []
dzdt_maxs = []
#
erosionrate_means = []
erosionrate_medians = []
erosionrate_2s = []
erosionrate_98s = []
erosionrate_mins = []
erosionrate_maxs = []
#
netdz_means = []
netdz_medians = []
netdz_2s = []
netdz_98s = []
netdz_mins = []
netdz_maxs = []
#
sediment_flux_gauge = []
sediment_flux_outlet = []

# Print space
print(' ')

# Initialize previous_zr
print('Starting time loop...')  
print(' ')
for ti in t:
      
    # Uplift topograpghy
    mg.at_node['bedrock__elevation'][mg.core_nodes] += U[mg.core_nodes] * dt    
    mg.at_node['topographic__elevation'][mg.core_nodes] = mg.at_node['bedrock__elevation'][mg.core_nodes] + mg.at_node['soil__depth'][mg.core_nodes]
    
    # Run one steps                       
    frr.run_one_step()                                    
    spr.run_one_step(dt)
    dfn.run_one_step(dt)
    eet.run_one_step(dt, uplift = Be10_Uplift * dt)

    # Calculate dz_dt and erosion rate metrics
    dbedelevdt = eet.return_dbedelevdt()
    dsoildepthdt = eet.return_dsoildepthdt()    
    dzdt = eet.return_dzdt()
    erosionrate = eet.return_erosionrate()
    netdz = mg.at_node['topographic__elevation'] - zr0
    
    # Correct for no_data_nodes
    z_adjust = np.copy(mg.at_node['topographic__elevation'])
    z_adjust[no_data_nodes] = nan
    dbedelevdt_adjust = np.copy(dbedelevdt)
    dbedelevdt_adjust[no_data_nodes] = nan
    dsoildepthdt_adjust = np.copy(dsoildepthdt)
    dsoildepthdt_adjust[no_data_nodes] = nan
    dzdt_adjust = np.copy(dzdt)
    dzdt_adjust[no_data_nodes] = nan
    erosionrate_adjust = np.copy(erosionrate)
    erosionrate_adjust[no_data_nodes] = nan
    netdz_adjust = np.copy(netdz)
    netdz_adjust[no_data_nodes] = nan
    
    # Append timestep stats
    timesteps = np.append(timesteps, timestep_integer)
    times = timesteps * dt
    #
    z_means = np.append(z_means, np.nanmean(z_adjust))
    z_medians = np.append(z_medians, np.nanmedian(z_adjust))
    z_2s = np.append(z_2s, np.nanpercentile(z_adjust, 2.3))
    z_98s = np.append(z_98s, np.nanpercentile(z_adjust, 97.7))
    z_mins = np.append(z_mins, np.nanmin(z_adjust))
    z_maxs = np.append(z_maxs, np.nanmax(z_adjust))
    #
    dbedelevdt_means = np.append(dbedelevdt_means, np.nanmean(dbedelevdt_adjust))
    dbedelevdt_medians = np.append(dbedelevdt_medians, np.nanmedian(dbedelevdt_adjust))
    dbedelevdt_2s = np.append(dbedelevdt_2s, np.nanpercentile(dbedelevdt_adjust, 2.3))
    dbedelevdt_98s = np.append(dbedelevdt_98s, np.nanpercentile(dbedelevdt_adjust, 97.7))
    dbedelevdt_mins = np.append(dbedelevdt_mins, np.nanmin(dbedelevdt_adjust))
    dbedelevdt_maxs = np.append(dbedelevdt_maxs, np.nanmax(dbedelevdt_adjust))
    #
    dsoildepthdt_means = np.append(dsoildepthdt_means, np.nanmean(dsoildepthdt_adjust))
    dsoildepthdt_medians = np.append(dsoildepthdt_medians, np.nanmedian(dsoildepthdt_adjust))
    dsoildepthdt_2s = np.append(dsoildepthdt_2s, np.nanpercentile(dsoildepthdt_adjust, 2.3))
    dsoildepthdt_98s = np.append(dsoildepthdt_98s, np.nanpercentile(dsoildepthdt_adjust, 97.7))
    dsoildepthdt_mins = np.append(dsoildepthdt_mins, np.nanmin(dsoildepthdt_adjust))
    dsoildepthdt_maxs = np.append(dsoildepthdt_maxs, np.nanmax(dsoildepthdt_adjust))
    #
    dzdt_means = np.append(dzdt_means, np.nanmean(dzdt_adjust))
    dzdt_medians = np.append(dzdt_medians, np.nanmedian(dzdt_adjust))
    dzdt_2s = np.append(dzdt_2s, np.nanpercentile(dzdt_adjust, 2.3))
    dzdt_98s = np.append(dzdt_98s, np.nanpercentile(dzdt_adjust, 97.7))
    dzdt_mins = np.append(dzdt_mins, np.nanmin(dzdt_adjust))
    dzdt_maxs = np.append(dzdt_maxs, np.nanmax(dzdt_adjust))
    #
    erosionrate_means = np.append(erosionrate_means, np.nanmean(erosionrate_adjust))
    erosionrate_medians = np.append(erosionrate_medians, np.nanmedian(erosionrate_adjust))
    erosionrate_2s = np.append(erosionrate_2s, np.nanpercentile(erosionrate_adjust, 2.3))
    erosionrate_98s = np.append(erosionrate_98s, np.nanpercentile(erosionrate_adjust, 97.7))
    erosionrate_mins = np.append(erosionrate_mins, np.nanmin(erosionrate_adjust))
    erosionrate_maxs = np.append(erosionrate_maxs, np.nanmax(erosionrate_adjust))
    #
    netdz_means = np.append(netdz_means, np.nanmean(netdz_adjust))
    netdz_medians = np.append(netdz_medians, np.nanmedian(netdz_adjust))
    netdz_2s = np.append(netdz_2s, np.nanpercentile(netdz_adjust, 2.3))
    netdz_98s = np.append(netdz_98s, np.nanpercentile(netdz_adjust, 97.7))
    netdz_mins = np.append(netdz_mins, np.nanmin(netdz_adjust))
    netdz_maxs = np.append(netdz_maxs, np.nanmax(netdz_adjust))

    # Append Fluxes
    # sediment_flux_gauge = np.append(sediment_flux_gauge, np.max(mg.at_node['sediment__flux'][gauge_radius_nodes]))
    sediment_flux_outlet = np.append(sediment_flux_outlet, np.max(mg.at_node['sediment__flux'][outlet_radius_nodes]))
    
    # Update time
    total_time += dt                                    
    Plot_Ticker += dt
    Export_DEM_Ticker += dt
    timestep_integer += 1
    total_time_r = np.round(total_time, decimals = 1)
    print(total_time_r, ' ', np.max(mg.at_node['sediment__flux'][outlet_radius_nodes]))

    # Plots
    if Plot_Ticker >= Plot_interval - 0.0000000001:
        print('Exporting figures... Please be patient!')
        
        # Plot__topographic__elevation
        if Plot__topographic__elevation == True:
            #
            if path.exists(str(Directory)+'/TerrainSandbox/topographic__elevation_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/topographic__elevation_Map')
            #
            drainage_area = mg.at_node['drainage_area']
            stream_nodes = np.where(drainage_area > min_drainage_area)[0]
            #
            plt.ioff()
            fig = plt.figure()       
            ax = imshowhs_grid(mg, "topographic__elevation", plot_type='Drape1', drape1=mg.at_node['topographic__elevation'],cmap='terrain',alpha=alpha,color_for_background='k',color_for_closed='k',allow_colorbar=True,cbar_loc="right",cbar_or='vertical',bbox_to_anchor=(1.05, 0.0, 0.03, 1.0),cbar_width="100%", cbar_height="100%", var_name='Elevation (m)', colorbar_label_x=1.5, colorbar_label_y=0.5, cbar_label_fontweight='regular', add_label_bbox=False)
            ax.set_title('$Year$=' + str(total_time_r), fontsize=12, loc='center', pad=20)
            formatter = ScalarFormatter()
            formatter.set_powerlimits((-3, 3))  # Switch to scientific notation if the range is <10^-3 or >10^3
            ax.xaxis.set_major_formatter(formatter)
            plt.tight_layout()
            if path.exists(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Full_NS') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Full_NS')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Full_NS/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            ax.plot(mg.x_of_node[stream_nodes], mg.y_of_node[stream_nodes], 'k.', markersize = 1.5)
            #
            if path.exists(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Full')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            if path.exists(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Zone_1') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Zone_1')
            ax.set_xlim([683140, 695270])
            ax.set_ylim([3427940, 3440170])  
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Zone_1/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Zone_2') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Zone_2')
            ax.set_xlim([664880, 682180])
            ax.set_ylim([3475450, 3492060])
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Zone_2/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)   
            #
            if path.exists(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Zone_3') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Zone_3')
            ax.set_xlim([681244, 686716])
            ax.set_ylim([3495674, 3501314])  
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/topographic__elevation_Map/Zone_3/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig) 
            #
            if path.exists(str(Directory)+'/TerrainSandbox/topographic__elevation_timeseries') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/topographic__elevation_timeseries')  
            fig = plt.figure()
            plt.plot(times, z_means, color = 'blue', zorder = 4) 
            plt.plot(times, z_medians, color = 'red', zorder = 4)
            plt.fill_between(times, z_2s, z_98s, alpha=0.50, color = 'blue', zorder = 3)
            plt.xlabel('Model year')
            plt.ylabel("Elevation (m)")
            plt.grid()
            plt.legend(['Mean', 'Median', "2σ"], loc='upper left', framealpha = 1)
            fig.savefig(str(Directory)+'/TerrainSandbox/topographic__elevation_timeseries/topographic__elevation_timeseries.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)
            
        # Plot__dbedelevdt
        if Plot__dbedelevdt == True:
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dbedelevdt_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dbedelevdt_Map')
            #
            plt.ioff()
            fig, ax = plt.subplots()
            drape1 = dbedelevdt
            cmap = 'PRGn'
            label = "d(bed elevation) dt⁻¹ (m yr⁻¹)"
            vmin = -np.percentile(np.abs(drape1), 99)
            vmax = np.percentile(np.abs(drape1), 99)
            im = imshowhs_grid(mg, "topographic__elevation", plot_type='Drape1', drape1=drape1, cmap=cmap, alpha=alpha, color_for_background='k', color_for_closed='k', allow_colorbar=False, cbar_loc="right", cbar_or='vertical', bbox_to_anchor=(1.05, 0.0, 0.03, 1.0), cbar_width="100%", cbar_height="100%", var_name='Bedrock elevation (m)', colorbar_label_x=1.5, colorbar_label_y=0.5, cbar_label_fontweight='regular', add_label_bbox=False, vmin = vmin, vmax = vmax)
            ticks = np.linspace(vmin, vmax, 7)            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])  # No data, just for colorbar
            cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
            cbar.set_label(label, fontsize=12)
            ax.set_title('$Year$=' + str(total_time_r), fontsize=12, loc='center', pad=20)
            formatter = ScalarFormatter()
            formatter.set_powerlimits((-3, 3))  # Switch to scientific notation if the range is <10^-3 or >10^3
            ax.xaxis.set_major_formatter(formatter)
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_notation_format))
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Full')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Zone_1') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Zone_1')
            ax.set_xlim([683140, 695270])
            ax.set_ylim([3427940, 3440170]) 
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Zone_1/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Zone_2') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Zone_2')
            ax.set_xlim([664880, 682180])
            ax.set_ylim([3475450, 3492060]) 
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Zone_2/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)   
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Zone_3') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Zone_3')
            ax.set_xlim([681244, 686716])
            ax.set_ylim([3495674, 3501314])   
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dbedelevdt_Map/Zone_3/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dbedelevdt_timeseries') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dbedelevdt_timeseries')  
            fig = plt.figure()
            plt.plot(times, dbedelevdt_means, color = 'blue', zorder = 4) 
            plt.plot(times, dbedelevdt_medians, color = 'red', zorder = 4)
            plt.fill_between(times, dbedelevdt_2s, dbedelevdt_98s, alpha=0.50, color = 'blue', zorder = 3)
            plt.xlabel('Model year')
            plt.ylabel("d(bed elevation) dt⁻¹ (m yr⁻¹)")
            plt.grid()
            plt.legend(['Mean', 'Median', "2σ"], loc='lower left', framealpha = 1)
            fig.savefig(str(Directory)+'/TerrainSandbox/dbedelevdt_timeseries/dbedelevdt_timeseries.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)
            
        # Plot__dsoildepthdt
        if Plot__dsoildepthdt == True:
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map')
            #
            plt.ioff()
            fig, ax = plt.subplots()
            drape1 = dsoildepthdt
            cmap = 'PiYG'
            label = "d(regolith depth) dt⁻¹ (m yr⁻¹)"
            vmin = -np.percentile(np.abs(drape1), 99)
            vmax = np.percentile(np.abs(drape1), 99)
            im = imshowhs_grid(mg, "topographic__elevation", plot_type='Drape1', drape1=drape1, cmap=cmap, alpha=alpha, color_for_background='k', color_for_closed='k', allow_colorbar=False, cbar_loc="right", cbar_or='vertical', bbox_to_anchor=(1.05, 0.0, 0.03, 1.0), cbar_width="100%", cbar_height="100%", var_name='Bedrock elevation (m)', colorbar_label_x=1.5, colorbar_label_y=0.5, cbar_label_fontweight='regular', add_label_bbox=False, vmin = vmin, vmax = vmax)
            ticks = np.linspace(vmin, vmax, 7)            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])  # No data, just for colorbar
            cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
            cbar.set_label(label, fontsize=12)
            ax.set_title('$Year$=' + str(total_time_r), fontsize=12, loc='center', pad=20)
            formatter = ScalarFormatter()
            formatter.set_powerlimits((-3, 3))  # Switch to scientific notation if the range is <10^-3 or >10^3
            ax.xaxis.set_major_formatter(formatter)      
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_notation_format))
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Full')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Zone_1') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Zone_1')
            ax.set_xlim([683140, 695270])
            ax.set_ylim([3427940, 3440170])  
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Zone_1/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Zone_2') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Zone_2')
            ax.set_xlim([664880, 682180])
            ax.set_ylim([3475450, 3492060]) 
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Zone_2/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)   
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Zone_3') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Zone_3')
            ax.set_xlim([681244, 686716])
            ax.set_ylim([3495674, 3501314])   
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dsoildepthdt_Map/Zone_3/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dsoildepthdt_timeseries') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dsoildepthdt_timeseries')  
            fig = plt.figure()
            plt.plot(times, dsoildepthdt_means, color = 'blue', zorder = 4) 
            plt.plot(times, dsoildepthdt_medians, color = 'red', zorder = 4)
            plt.fill_between(times, dsoildepthdt_2s, dsoildepthdt_98s, alpha=0.50, color = 'blue', zorder = 3)
            plt.xlabel('Model year')
            plt.ylabel("d(regolith depth) dt⁻¹ (m yr⁻¹)")
            plt.grid()
            plt.legend(['Mean', 'Median', "2σ"], loc='lower right', framealpha = 1)
            fig.savefig(str(Directory)+'/TerrainSandbox/dsoildepthdt_timeseries/dsoildepthdt_timeseries.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)

        # Plot__dzdt
        if Plot__dzdt == True:
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dzdt_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dzdt_Map')
            #
            # ax = imshowhs_grid(mg, mg.at_node['topographic__elevation'], plot_name = '$Year$='+str(total_time_r), var_name = "dz dt⁻¹", plot_type='Drape1', drape1 = dzdt, cmap='seismic_r', alpha = alpha, color_for_background = 'k', color_for_closed = 'k', allow_colorbar = True, vmin = -np.percentile(np.abs(dzdt), 99), vmax = np.percentile(np.abs(dzdt), 99))
            plt.ioff()
            fig, ax = plt.subplots()
            drape1 = dzdt
            cmap = 'seismic_r'
            label = "dz dt⁻¹ (m yr⁻¹)" 
            vmin = -np.percentile(np.abs(drape1), 99)
            vmax = np.percentile(np.abs(drape1), 99)
            im = imshowhs_grid(mg, "topographic__elevation", plot_type='Drape1', drape1=drape1, cmap=cmap, alpha=alpha, color_for_background='k', color_for_closed='k', allow_colorbar=False, cbar_loc="right", cbar_or='vertical', bbox_to_anchor=(1.05, 0.0, 0.03, 1.0), cbar_width="100%", cbar_height="100%", var_name='Bedrock elevation (m)', colorbar_label_x=1.5, colorbar_label_y=0.5, cbar_label_fontweight='regular', add_label_bbox=False, vmin = vmin, vmax = vmax)
            ticks = np.linspace(vmin, vmax, 7)            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])  # No data, just for colorbar
            cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
            cbar.set_label(label, fontsize=12)
            ax.set_title('$Year$=' + str(total_time_r), fontsize=12, loc='center', pad=20)
            formatter = ScalarFormatter()
            formatter.set_powerlimits((-3, 3))  # Switch to scientific notation if the range is <10^-3 or >10^3
            ax.xaxis.set_major_formatter(formatter)    
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_notation_format))
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dzdt_Map/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dzdt_Map/Full')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dzdt_Map/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dzdt_Map/Zone_1') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dzdt_Map/Zone_1')
            ax.set_xlim([683140, 695270])
            ax.set_ylim([3427940, 3440170])  
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dzdt_Map/Zone_1/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dzdt_Map/Zone_2') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dzdt_Map/Zone_2')
            ax.set_xlim([664880, 682180])
            ax.set_ylim([3475450, 3492060])
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dzdt_Map/Zone_2/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)   
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dzdt_Map/Zone_3') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dzdt_Map/Zone_3')
            ax.set_xlim([681244, 686716])
            ax.set_ylim([3495674, 3501314])   
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/dzdt_Map/Zone_3/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/dzdt_timeseries') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/dzdt_timeseries')  
            fig = plt.figure()
            plt.plot(times, dzdt_means, color = 'blue', zorder = 4) 
            plt.plot(times, dzdt_medians, color = 'red', zorder = 4)
            plt.fill_between(times, dzdt_2s, dzdt_98s, alpha=0.50, color = 'blue', zorder = 3)
            plt.xlabel('Model year')
            plt.ylabel("dz dt⁻¹ (m yr⁻¹)")
            plt.grid()
            plt.legend(['Mean', 'Median', "2σ"], loc='lower right', framealpha = 1)
            fig.savefig(str(Directory)+'/TerrainSandbox/dzdt_timeseries/dzdt_timeseries.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)
            
        # Plot__erosionrate
        if Plot__erosionrate == True:
            #
            if path.exists(str(Directory)+'/TerrainSandbox/erosionrate_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/erosionrate_Map')
            #        
            # ax = imshowhs_grid(mg, mg.at_node['topographic__elevation'], plot_name = '$Year$='+str(total_time_r), var_name = "erosion rate", plot_type='Drape1', drape1 = dzdt, cmap='BrBG', alpha = alpha, color_for_background = 'k', color_for_closed = 'k', allow_colorbar = True, vmin = -np.percentile(np.abs(erosionrate), 99), vmax = np.percentile(np.abs(erosionrate), 99))
            plt.ioff()
            fig, ax = plt.subplots()
            drape1 = erosionrate
            cmap = 'BrBG_r'
            label = "Erosion rate (m yr⁻¹)" 
            vmin = -np.percentile(np.abs(drape1), 99)
            vmax = np.percentile(np.abs(drape1), 99)
            im = imshowhs_grid(mg, "topographic__elevation", plot_type='Drape1', drape1=drape1, cmap=cmap, alpha=alpha, color_for_background='k', color_for_closed='k', allow_colorbar=False, cbar_loc="right", cbar_or='vertical', bbox_to_anchor=(1.05, 0.0, 0.03, 1.0), cbar_width="100%", cbar_height="100%", var_name='Bedrock elevation (m)', colorbar_label_x=1.5, colorbar_label_y=0.5, cbar_label_fontweight='regular', add_label_bbox=False, vmin = vmin, vmax = vmax)
            ticks = np.linspace(vmin, vmax, 7)            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])  # No data, just for colorbar
            cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
            cbar.set_label(label, fontsize=12)
            ax.set_title('$Year$=' + str(total_time_r), fontsize=12, loc='center', pad=20)
            formatter = ScalarFormatter()
            formatter.set_powerlimits((-3, 3))  # Switch to scientific notation if the range is <10^-3 or >10^3
            ax.xaxis.set_major_formatter(formatter)  
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_notation_format))
            #
            if path.exists(str(Directory)+'/TerrainSandbox/erosionrate_Map/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/erosionrate_Map/Full')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/erosionrate_Map/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            if path.exists(str(Directory)+'/TerrainSandbox/erosionrate_Map/Zone_1') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/erosionrate_Map/Zone_1')
            ax.set_xlim([683140, 695270])
            ax.set_ylim([3427940, 3440170])  
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/erosionrate_Map/Zone_1/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/erosionrate_Map/Zone_2') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/erosionrate_Map/Zone_2')
            ax.set_xlim([664880, 682180])
            ax.set_ylim([3475450, 3492060]) 
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/erosionrate_Map/Zone_2/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)   
            #
            if path.exists(str(Directory)+'/TerrainSandbox/erosionrate_Map/Zone_3') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/erosionrate_Map/Zone_3')
            ax.set_xlim([681244, 686716])
            ax.set_ylim([3495674, 3501314])    
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/erosionrate_Map/Zone_3/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/erosionrate_timeseries') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/erosionrate_timeseries')  
            fig = plt.figure()
            plt.plot(times, erosionrate_means, color = 'blue', zorder = 4) 
            plt.plot(times, erosionrate_medians, color = 'red', zorder = 4)
            plt.fill_between(times, erosionrate_2s, erosionrate_98s, alpha=0.50, color = 'blue', zorder = 3)
            plt.xlabel('Model year')
            plt.ylabel("erosion rate (m yr⁻¹)")
            plt.grid()
            plt.legend(['Mean', 'Median', "2σ"], loc='upper right', framealpha = 1)
            fig.savefig(str(Directory)+'/TerrainSandbox/erosionrate_timeseries/erosionrate_timeseries.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)

        # Plot__netdz
        if Plot__netdz == True:
            #
            if path.exists(str(Directory)+'/TerrainSandbox/netdz_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/netdz_Map')
            #     
            # ax = imshowhs_grid(mg, "topographic__elevation", plot_name = '$Year$='+str(total_time_r), var_name = 'Net elevational change (m)', plot_type='Drape1', drape1 = netdz, cmap='seismic_r', alpha = alpha, color_for_background = 'k', color_for_closed = 'k', allow_colorbar = True, vmin = -np.percentile(np.abs(netdz), 99), vmax = np.percentile(np.abs(netdz), 99))
            plt.ioff()
            fig, ax = plt.subplots()
            drape1 = netdz
            cmap = 'seismic_r'
            label = "Net elevational change (m)" 
            vmin = -np.percentile(np.abs(drape1), 99)
            vmax = np.percentile(np.abs(drape1), 99)
            im = imshowhs_grid(mg, "topographic__elevation", plot_type='Drape1', drape1=drape1, cmap=cmap, alpha=alpha, color_for_background='k', color_for_closed='k', allow_colorbar=False, cbar_loc="right", cbar_or='vertical', bbox_to_anchor=(1.05, 0.0, 0.03, 1.0), cbar_width="100%", cbar_height="100%", var_name='Bedrock elevation (m)', colorbar_label_x=1.5, colorbar_label_y=0.5, cbar_label_fontweight='regular', add_label_bbox=False, vmin = vmin, vmax = vmax)
            ticks = np.linspace(vmin, vmax, 7)            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])  # No data, just for colorbar
            cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
            cbar.set_label(label, fontsize=12)
            ax.set_title('$Year$=' + str(total_time_r), fontsize=12, loc='center', pad=20)
            formatter = ScalarFormatter()
            formatter.set_powerlimits((-3, 3))  # Switch to scientific notation if the range is <10^-3 or >10^3
            ax.xaxis.set_major_formatter(formatter) 
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_notation_format))
            #
            if path.exists(str(Directory)+'/TerrainSandbox/netdz_Map/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/netdz_Map/Full')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/netdz_Map/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            if path.exists(str(Directory)+'/TerrainSandbox/netdz_Map/Zone_1') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/netdz_Map/Zone_1')
            ax.set_xlim([683140, 695270])
            ax.set_ylim([3427940, 3440170])  
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/netdz_Map/Zone_1/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)  
            #
            if path.exists(str(Directory)+'/TerrainSandbox/netdz_Map/Zone_2') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/netdz_Map/Zone_2')
            ax.set_xlim([664880, 682180])
            ax.set_ylim([3475450, 3492060])
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/netdz_Map/Zone_2/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)   
            #
            if path.exists(str(Directory)+'/TerrainSandbox/netdz_Map/Zone_3') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/netdz_Map/Zone_3')
            ax.set_xlim([681244, 686716])
            ax.set_ylim([3495674, 3501314])   
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/netdz_Map/Zone_3/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig) 
            #
            if path.exists(str(Directory)+'/TerrainSandbox/netdz_timeseries') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/netdz_timeseries')  
            fig = plt.figure()
            plt.plot(times, netdz_means, color = 'blue', zorder = 4) 
            plt.plot(times, netdz_medians, color = 'red', zorder = 4)
            plt.fill_between(times, netdz_2s, netdz_98s, alpha=0.50, color = 'blue', zorder = 3)
            plt.xlabel('Model year')
            plt.ylabel("Net elevational change (m)")
            plt.grid()
            plt.legend(['Mean', 'Median', "2σ"], loc='lower left', framealpha = 1)
            fig.savefig(str(Directory)+'/TerrainSandbox/netdz_timeseries/netdz_timeseries.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)           
            
        # Plot__sediment__flux
        if Plot__sediment__flux == True:
            if path.exists(str(Directory)+'/TerrainSandbox/sediment__flux_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/sediment__flux_Map')
            plt.ioff()
            fig = plt.figure(1)         
            imshow_grid(mg, 'sediment__flux', grid_units=('m', 'm'), var_name="Sediment flux (m³ yr⁻¹)", cmap='terrain', allow_colorbar=True)
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/sediment__flux_Map/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            plt.close(fig)
        
        # # sediment__flux_timeseries_gauge
        # if sediment__flux_timeseries_gauge == True:
        #     if path.exists(str(Directory)+'/TerrainSandbox/sediment__flux_timeseries_gauge') == False:
        #         os.mkdir(str(Directory)+'/TerrainSandbox/sediment__flux_timeseries_gauge')
        #     plt.ioff()
        #     fig, ax1 = plt.subplots()
        #     ax1.plot(times, sediment_flux_gauge, 'k-', zorder = 2)
        #     ax1.set_xlabel('Model year')
        #     ax1.set_ylabel('Sediment flux at gauge (m³ yr⁻¹)')
        #     ax1.grid(linestyle='--')
        #     y = (times * 0) + (Be10_Uplift * np.max(mg.at_node['drainage_area'][gauge_radius_nodes]))
        #     yl = (times * 0) + (Be10_Uplift_min * np.max(mg.at_node['drainage_area'][gauge_radius_nodes]))
        #     yh = (times * 0) + (Be10_Uplift_max * np.max(mg.at_node['drainage_area'][gauge_radius_nodes]))
        #     ax1.plot(times, y, 'b--', zorder = 2)
        #     ax1.fill_between(times, yl, yh, alpha=0.50, color = 'blue', zorder = 1)
        #     ax2 = ax1.twinx()
        #     ymin, ymax = ax1.get_ylim()
        #     ax2.set_ylim(ymin / np.max(mg.at_node['drainage_area'][gauge_radius_nodes]), ymax / np.max(mg.at_node['drainage_area'][gauge_radius_nodes]))
        #     ax2.set_ylabel('Equivalent MCER (m yr⁻¹)')
        #     ax1.legend(['Model data at gauge', '¹⁰Be-implied MCER', "¹⁰Be-implied ± 1σ"], loc='lower right', framealpha=1)            
        #     plt.tight_layout()
        #     fig.savefig(str(Directory)+'/TerrainSandbox/sediment__flux_timeseries_gauge/sediment__flux_timeseries_gauge.' + Export_format,  format = Export_format, dpi=dpi)
        #     plt.close(fig) 
            
        # sediment__flux_timeseries_outlet
        if sediment__flux_timeseries_outlet == True:
            if path.exists(str(Directory)+'/TerrainSandbox/sediment__flux_timeseries_outlet') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/sediment__flux_timeseries_outlet')
            plt.ioff()
            fig, ax1 = plt.subplots()
            ax1.plot(times, sediment_flux_outlet, 'k-', zorder = 2)
            ax1.set_xlabel('Model year')
            ax1.set_ylabel('Sediment flux at outlet (m³ yr⁻¹)')
            ax1.grid(linestyle='--')
            y = (times * 0) + (Be10_Uplift * np.max(mg.at_node['drainage_area'][outlet_radius_nodes]))
            yl = (times * 0) + (Be10_Uplift_min * np.max(mg.at_node['drainage_area'][outlet_radius_nodes]))
            yh = (times * 0) + (Be10_Uplift_max * np.max(mg.at_node['drainage_area'][outlet_radius_nodes]))
            ax1.plot(times, y, 'b--', zorder = 2)
            ax1.fill_between(times, yl, yh, alpha=0.50, color = 'blue', zorder = 1)
            ax2 = ax1.twinx()
            ymin, ymax = ax1.get_ylim()
            ax2.set_ylim(ymin / np.max(mg.at_node['drainage_area'][outlet_radius_nodes]), ymax / np.max(mg.at_node['drainage_area'][outlet_radius_nodes]))
            ax2.set_ylabel('Equivalent MCER (m yr⁻¹)')
            ax1.legend(['Model data at outlet', '¹⁰Be-implied MCER', "¹⁰Be-implied ± 1σ"], loc='lower right', framealpha=1)            
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/sediment__flux_timeseries_outlet/sediment__flux_timeseries_outlet.' + Export_format,  format = Export_format, dpi=dpi)
            plt.close(fig) 
        
        # Plot__Seg1
        if Plot__Seg1 == True:
            
            # Extract seg1
            receivers = mg.at_node["flow__receiver_node"]
            start_node = seg1_head
            seg1_nodes_ti = [start_node]
            current_node = start_node
            while True:
                next_node = receivers[current_node]
                if next_node == current_node:
                    break  # reached the outlet
                seg1_nodes_ti.append(next_node)
                current_node = next_node
            seg1_nodes_ti = np.flip(np.array(seg1_nodes_ti))
            #
            # Find overlapping nodes between current and initial profiles
            seg1_nodes_use = []
            seg1_distance_use = []
            for i in np.arange(0, len(seg1_nodes)):
                if seg1_nodes[i] in seg1_nodes_ti:
                    seg1_nodes_use = np.append(seg1_nodes_use, seg1_nodes[i])
                    seg1_distance_use = np.append(seg1_distance_use, seg1_distance[i])
            seg1_nodes_use = seg1_nodes_use.astype(int)
            #
            # Find whether seg1_nodes_use are depositing or eroding
            seg1_nodes_use_ed = np.copy(seg1_nodes_use).astype(float)
            for i in np.arange(0, len(seg1_nodes_use)):
                if erosionrate[seg1_nodes_use[i]] > 0:
                    seg1_nodes_use_ed[i] = -1
                elif erosionrate[seg1_nodes_use[i]] < 0:
                    seg1_nodes_use_ed[i] = 1
                elif erosionrate[seg1_nodes_use[i]] == 1:
                    seg1_nodes_use_ed[i] = 0
                    
            # Plot seg1
            #
            if path.exists(str(Directory)+'/TerrainSandbox/Seg1') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Seg1')
            plt.ioff()
            fig = plt.figure()   
            plt.plot(seg1_distance_use / 1000, mg.at_node['topographic__elevation'][seg1_nodes_use], '.', linewidth = 0.5, color = 'orange')
            plt.plot(seg1_distance / 1000, zr0[seg1_nodes], '--', linewidth = 0.5, color = 'orange')
            plt.plot(seg1_distance_use / 1000, mg.at_node['bedrock__elevation'][seg1_nodes_use], 'k.', linewidth = 0.5)        
            plt.plot(seg1_distance / 1000, br0[seg1_nodes], 'k--', linewidth = 0.5)
            for i in np.arange(0, len(seg1_nodes_use_ed)):
                if seg1_nodes_use_ed[i] == 1:
                    plt.plot(seg1_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg1_nodes_use[i]], '.', linewidth = 0.5, color = 'green')
                elif seg1_nodes_use_ed[i] == -1:
                    plt.plot(seg1_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg1_nodes_use[i]], '.', linewidth = 0.5, color = 'red')
                if dbedelevdt[seg1_nodes_use[i]] < 0:
                    plt.plot(seg1_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg1_nodes_use[i]], '.', linewidth = 0.5, color = 'purple')        
            plt.grid()
            plt.xlabel('Distance (km)')
            plt.ylabel('Elevation (m)')
            plt.title('$Year$='+str(total_time_r))
            plt.legend(['Current topographic elevation', 'Initial topographic elevation', 'Current bedrock elevation', 'Initial bedrock elevation'], loc = 'upper left', fontsize=6)
            #
            # Plot full
            if path.exists(str(Directory)+'/TerrainSandbox/Seg1/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Seg1/Full')
            plt.xlim(0, 56)
            plt.ylim(0, 80)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Seg1/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            # Zone 1
            if path.exists(str(Directory)+'/TerrainSandbox/Seg1/Zone_1') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Seg1/Zone_1')
            plt.xlim(0, 15)
            plt.ylim(10, 40)
            fig.savefig(str(Directory)+'/TerrainSandbox/Seg1/Zone_1/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)                
                
        # Plot__Seg2
        if Plot__Seg2 == True:                
                
            # Extract seg2
            receivers = mg.at_node["flow__receiver_node"]
            start_node = seg2_head
            seg2_nodes_ti = [start_node]
            current_node = start_node
            while True:
                next_node = receivers[current_node]
                if next_node == current_node:
                    break  # reached the outlet
                seg2_nodes_ti.append(next_node)
                current_node = next_node
            seg2_nodes_ti = np.flip(np.array(seg2_nodes_ti))
            #
            # Find overlapping nodes between current and initial profiles
            seg2_nodes_use = []
            seg2_distance_use = []
            for i in np.arange(0, len(seg2_nodes)):
                if seg2_nodes[i] in seg2_nodes_ti:
                    seg2_nodes_use = np.append(seg2_nodes_use, seg2_nodes[i])
                    seg2_distance_use = np.append(seg2_distance_use, seg2_distance[i])
            seg2_nodes_use = seg2_nodes_use.astype(int)
            #
            # Find whether seg1_nodes_use are depositing or eroding
            seg2_nodes_use_ed = np.copy(seg2_nodes_use).astype(float)
            for i in np.arange(0, len(seg2_nodes_use)):
                if erosionrate[seg2_nodes_use[i]] > 0:
                    seg2_nodes_use_ed[i] = -1
                elif erosionrate[seg2_nodes_use[i]] < 0:
                    seg2_nodes_use_ed[i] = 1
                elif erosionrate[seg2_nodes_use[i]] == 1:
                    seg2_nodes_use_ed[i] = 0
                    
            # Plot seg2
            #
            if path.exists(str(Directory)+'/TerrainSandbox/Seg2') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Seg2')
            plt.ioff()
            fig = plt.figure()   
            plt.plot(seg2_distance_use / 1000, mg.at_node['topographic__elevation'][seg2_nodes_use], '.', linewidth = 0.5, color = 'orange')
            plt.plot(seg2_distance / 1000, zr0[seg2_nodes], '--', linewidth = 0.5, color = 'orange')
            plt.plot(seg2_distance_use / 1000, mg.at_node['bedrock__elevation'][seg2_nodes_use], 'k.', linewidth = 0.5)        
            plt.plot(seg2_distance / 1000, br0[seg2_nodes], 'k--', linewidth = 0.5)
            for i in np.arange(0, len(seg2_nodes_use_ed)):
                if seg2_nodes_use_ed[i] == 1:
                    plt.plot(seg2_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg2_nodes_use[i]], '.', linewidth = 0.5, color = 'green')
                elif seg2_nodes_use_ed[i] == -1:
                    plt.plot(seg2_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg2_nodes_use[i]], '.', linewidth = 0.5, color = 'red')
                if dbedelevdt[seg2_nodes_use[i]] < 0:
                    plt.plot(seg2_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg2_nodes_use[i]], '.', linewidth = 0.5, color = 'purple')        
            plt.grid()
            plt.xlabel('Distance (km)')
            plt.ylabel('Elevation (m)')
            plt.title('$Year$='+str(total_time_r))
            plt.legend(['Current topographic elevation', 'Initial topographic elevation', 'Current bedrock elevation', 'Initial bedrock elevation'], loc = 'upper left', fontsize=6)
            #
            # Plot full
            if path.exists(str(Directory)+'/TerrainSandbox/Seg2/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Seg2/Full')
            plt.xlim(0, 125)
            plt.ylim(0, 130)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Seg2/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            # Zone 2
            if path.exists(str(Directory)+'/TerrainSandbox/Seg2/Zone_2') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Seg2/Zone_2')
            plt.xlim(61, 99)
            plt.ylim(10, 90)
            fig.savefig(str(Directory)+'/TerrainSandbox/Seg2/Zone_2/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)                    
                    
        # Plot__Seg3
        if Plot__Seg3 == True:                     
                    
            # Extract seg3
            receivers = mg.at_node["flow__receiver_node"]
            start_node = seg3_head
            seg3_nodes_ti = [start_node]
            current_node = start_node
            while True:
                next_node = receivers[current_node]
                if next_node == current_node:
                    break  # reached the outlet
                seg3_nodes_ti.append(next_node)
                current_node = next_node
            seg3_nodes_ti = np.flip(np.array(seg3_nodes_ti))
            #
            # Find overlapping nodes between current and initial profiles
            seg3_nodes_use = []
            seg3_distance_use = []
            for i in np.arange(0, len(seg3_nodes)):
                if seg3_nodes[i] in seg3_nodes_ti:
                    seg3_nodes_use = np.append(seg3_nodes_use, seg3_nodes[i])
                    seg3_distance_use = np.append(seg3_distance_use, seg3_distance[i])
            seg3_nodes_use = seg3_nodes_use.astype(int)
            #
            # Find whether seg3_nodes_use are depositing or eroding
            seg3_nodes_use_ed = np.copy(seg3_nodes_use).astype(float)
            for i in np.arange(0, len(seg3_nodes_use)):
                if erosionrate[seg3_nodes_use[i]] > 0:
                    seg3_nodes_use_ed[i] = -1
                elif erosionrate[seg3_nodes_use[i]] < 0:
                    seg3_nodes_use_ed[i] = 1
                elif erosionrate[seg3_nodes_use[i]] == 1:
                    seg3_nodes_use_ed[i] = 0            

            # Plot seg3
            #
            if path.exists(str(Directory)+'/TerrainSandbox/Seg3') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Seg3')
            plt.ioff()
            fig = plt.figure()   
            plt.plot(seg3_distance_use / 1000, mg.at_node['topographic__elevation'][seg3_nodes_use], '.', linewidth = 0.5, color = 'orange')
            plt.plot(seg3_distance / 1000, zr0[seg3_nodes], '--', linewidth = 0.5, color = 'orange')
            plt.plot(seg3_distance_use / 1000, mg.at_node['bedrock__elevation'][seg3_nodes_use], 'k.', linewidth = 0.5)        
            plt.plot(seg3_distance / 1000, br0[seg3_nodes], 'k--', linewidth = 0.5)
            for i in np.arange(0, len(seg3_nodes_use_ed)):
                if seg3_nodes_use_ed[i] == 1:
                    plt.plot(seg3_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg3_nodes_use[i]], '.', linewidth = 0.5, color = 'green')
                elif seg3_nodes_use_ed[i] == -1:
                    plt.plot(seg3_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg3_nodes_use[i]], '.', linewidth = 0.5, color = 'red')
                if dbedelevdt[seg3_nodes_use[i]] < 0:
                    plt.plot(seg3_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg3_nodes_use[i]], '.', linewidth = 0.5, color = 'purple')        
            plt.grid()
            plt.xlabel('Distance (km)')
            plt.ylabel('Elevation (m)')
            plt.title('$Year$='+str(total_time_r))
            plt.legend(['Current topographic elevation', 'Initial topographic elevation', 'Current bedrock elevation', 'Initial bedrock elevation'], loc = 'upper left', fontsize=6)
            #
            # Plot full
            if path.exists(str(Directory)+'/TerrainSandbox/Seg3/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Seg3/Full')
            plt.xlim(0, 90)
            plt.ylim(0, 45)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Seg3/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            # Zone 3
            if path.exists(str(Directory)+'/TerrainSandbox/Seg3/Zone_3') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Seg3/Zone_3')
            plt.xlim(83, 88)
            plt.ylim(15, 45)
            fig.savefig(str(Directory)+'/TerrainSandbox/Seg3/Zone_3/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)    

        # Plot__Segs_combined
        if Plot__Segs_combined == True: 
            #
            # Plot Segs_combined
            if path.exists(str(Directory)+'/TerrainSandbox/Segs_combined') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Segs_combined')
            #
            plt.ioff()
            fig = plt.figure() 
            #
            plt.plot(seg1_distance_use / 1000, mg.at_node['topographic__elevation'][seg1_nodes_use], '.', linewidth = 0.5, color = 'orange')
            plt.plot(seg1_distance / 1000, zr0[seg1_nodes], '--', linewidth = 0.5, color = 'orange')
            plt.plot(seg1_distance_use / 1000, mg.at_node['bedrock__elevation'][seg1_nodes_use], 'k.', linewidth = 0.5)        
            plt.plot(seg1_distance / 1000, br0[seg1_nodes], 'k--', linewidth = 0.5)
            #
            plt.plot(seg2_distance_use / 1000, mg.at_node['topographic__elevation'][seg2_nodes_use], '.', linewidth = 0.5, color = 'orange')
            plt.plot(seg2_distance / 1000, zr0[seg2_nodes], '--', linewidth = 0.5, color = 'orange')
            plt.plot(seg2_distance_use / 1000, mg.at_node['bedrock__elevation'][seg2_nodes_use], 'k.', linewidth = 0.5)        
            plt.plot(seg2_distance / 1000, br0[seg2_nodes], 'k--', linewidth = 0.5)
            #
            plt.plot(seg3_distance_use / 1000, mg.at_node['topographic__elevation'][seg3_nodes_use], '.', linewidth = 0.5, color = 'orange')
            plt.plot(seg3_distance / 1000, zr0[seg3_nodes], '--', linewidth = 0.5, color = 'orange')
            plt.plot(seg3_distance_use / 1000, mg.at_node['bedrock__elevation'][seg3_nodes_use], 'k.', linewidth = 0.5)        
            plt.plot(seg3_distance / 1000, br0[seg3_nodes], 'k--', linewidth = 0.5)
            #
            for i in np.arange(0, len(seg1_nodes_use_ed)):
                if seg1_nodes_use_ed[i] == 1:
                    plt.plot(seg1_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg1_nodes_use[i]], '.', linewidth = 0.5, color = 'green')
                elif seg1_nodes_use_ed[i] == -1:
                    plt.plot(seg1_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg1_nodes_use[i]], '.', linewidth = 0.5, color = 'red')
                if dbedelevdt[seg1_nodes_use[i]] < 0:
                    plt.plot(seg1_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg1_nodes_use[i]], '.', linewidth = 0.5, color = 'purple')        
            #
            for i in np.arange(0, len(seg2_nodes_use_ed)):
                if seg2_nodes_use_ed[i] == 1:
                    plt.plot(seg2_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg2_nodes_use[i]], '.', linewidth = 0.5, color = 'green')
                elif seg2_nodes_use_ed[i] == -1:
                    plt.plot(seg2_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg2_nodes_use[i]], '.', linewidth = 0.5, color = 'red')
                if dbedelevdt[seg2_nodes_use[i]] < 0:
                    plt.plot(seg2_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg2_nodes_use[i]], '.', linewidth = 0.5, color = 'purple')        
            #
            for i in np.arange(0, len(seg3_nodes_use_ed)):
                if seg3_nodes_use_ed[i] == 1:
                    plt.plot(seg3_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg3_nodes_use[i]], '.', linewidth = 0.5, color = 'green')
                elif seg3_nodes_use_ed[i] == -1:
                    plt.plot(seg3_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg3_nodes_use[i]], '.', linewidth = 0.5, color = 'red')
                if dbedelevdt[seg3_nodes_use[i]] < 0:
                    plt.plot(seg3_distance_use[i] / 1000, mg.at_node['topographic__elevation'][seg3_nodes_use[i]], '.', linewidth = 0.5, color = 'purple')        
            plt.grid()
            plt.xlabel('Distance (km)')
            plt.ylabel('Elevation (m)')
            plt.title('$Year$='+str(total_time_r))
            plt.legend(['Current topographic elevation', 'Initial topographic elevation', 'Current bedrock elevation', 'Initial bedrock elevation'], loc = 'upper left', fontsize=6)
            #
            # Plot full
            if path.exists(str(Directory)+'/TerrainSandbox/Segs_combined/Full') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Segs_combined/Full')
            plt.xlim(0, 125)
            plt.ylim(0, 130)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Segs_combined/Full/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            #
            plt.close(fig)   

        # Plot__xs_A
        if Plot__xs_A == True: 
            #
            # xs_A
            if path.exists(str(Directory)+'/TerrainSandbox/xs_A') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/xs_A')
            plt.ioff()
            fig = plt.figure() 
            #
            plt.plot(xs_A_distance, mg.at_node['topographic__elevation'][xs_A_nodes], '.', linewidth = 0.5, color = 'orange')
            plt.plot(xs_A_distance, zr0[xs_A_nodes], '--', linewidth = 0.5, color = 'orange')
            plt.plot(xs_A_distance, mg.at_node['bedrock__elevation'][xs_A_nodes], 'k.', linewidth = 0.5)
            plt.plot(xs_A_distance, br0[xs_A_nodes], 'k--', linewidth = 0.5)
            #
            for i in np.arange(0, len(xs_A_distance)):
                if dzdt[xs_A_nodes[i]] > 0:
                    plt.plot(xs_A_distance[i], mg.at_node['topographic__elevation'][xs_A_nodes[i]], '.', linewidth = 0.5, color = 'green')
                elif dzdt[xs_A_nodes[i]] < 0:
                    plt.plot(xs_A_distance[i], mg.at_node['topographic__elevation'][xs_A_nodes[i]], '.', linewidth = 0.5, color = 'red')
                if dbedelevdt[xs_A_nodes[i]] < 0:
                    plt.plot(xs_A_distance[i], mg.at_node['topographic__elevation'][xs_A_nodes[i]], '.', linewidth = 0.5, color = 'purple')
            #
            plt.ylim(10, 50)
            plt.grid()
            plt.xlabel('Distance (m)')
            plt.ylabel('Elevation (m)')
            plt.title('$Year$='+str(total_time_r))
            plt.legend(['Current topographic elevation', 'Initial topographic elevation', 'Current bedrock elevation', 'Initial bedrock elevation'], loc = 'upper left', fontsize=6)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/xs_A/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            plt.close(fig) 
            
        # Plot__xs_B
        if Plot__xs_B == True: 
            #        
            # xs_B
            if path.exists(str(Directory)+'/TerrainSandbox/xs_B') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/xs_B')
            plt.ioff()
            fig = plt.figure() 
            #
            plt.plot(xs_B_distance, mg.at_node['topographic__elevation'][xs_B_nodes], '.', linewidth = 0.5, color = 'orange')
            plt.plot(xs_B_distance, zr0[xs_B_nodes], '--', linewidth = 0.5, color = 'orange')
            plt.plot(xs_B_distance, mg.at_node['bedrock__elevation'][xs_B_nodes], 'k.', linewidth = 0.5)
            plt.plot(xs_B_distance, br0[xs_B_nodes], 'k--', linewidth = 0.5)
            #
            for i in np.arange(0, len(xs_B_distance)):
                if dzdt[xs_B_nodes[i]] > 0:
                    plt.plot(xs_B_distance[i], mg.at_node['topographic__elevation'][xs_B_nodes[i]], '.', linewidth = 0.5, color = 'green')
                elif dzdt[xs_B_nodes[i]] < 0:
                    plt.plot(xs_B_distance[i], mg.at_node['topographic__elevation'][xs_B_nodes[i]], '.', linewidth = 0.5, color = 'red')
                if dbedelevdt[xs_B_nodes[i]] < 0:
                    plt.plot(xs_B_distance[i], mg.at_node['topographic__elevation'][xs_B_nodes[i]], '.', linewidth = 0.5, color = 'purple')
            #
            plt.ylim(35, 120)
            plt.grid()
            plt.xlabel('Distance (m)')
            plt.ylabel('Elevation (m)')
            plt.title('$Year$='+str(total_time_r))
            plt.legend(['Current topographic elevation', 'Initial topographic elevation', 'Current bedrock elevation', 'Initial bedrock elevation'], loc = 'upper left', fontsize=6)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/xs_B/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            plt.close(fig) 

        # Plot__xs_C
        if Plot__xs_C == True: 
            #     
            # xs_C
            if path.exists(str(Directory)+'/TerrainSandbox/xs_C') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/xs_C')
            plt.ioff()
            fig = plt.figure() 
            #
            plt.plot(xs_C_distance, mg.at_node['topographic__elevation'][xs_C_nodes], '.', linewidth = 0.5, color = 'orange')
            plt.plot(xs_C_distance, zr0[xs_C_nodes], '--', linewidth = 0.5, color = 'orange')
            plt.plot(xs_C_distance, mg.at_node['bedrock__elevation'][xs_C_nodes], 'k.', linewidth = 0.5)
            plt.plot(xs_C_distance, br0[xs_C_nodes], 'k--', linewidth = 0.5)
            #
            for i in np.arange(0, len(xs_C_distance)):
                if dzdt[xs_C_nodes[i]] > 0:
                    plt.plot(xs_C_distance[i], mg.at_node['topographic__elevation'][xs_C_nodes[i]], '.', linewidth = 0.5, color = 'green')
                elif dzdt[xs_C_nodes[i]] < 0:
                    plt.plot(xs_C_distance[i], mg.at_node['topographic__elevation'][xs_C_nodes[i]], '.', linewidth = 0.5, color = 'red')
                if dbedelevdt[xs_C_nodes[i]] < 0:
                    plt.plot(xs_C_distance[i], mg.at_node['topographic__elevation'][xs_C_nodes[i]], '.', linewidth = 0.5, color = 'purple')
            #
            plt.ylim(15, 80)
            plt.grid()
            plt.xlabel('Distance (m)')
            plt.ylabel('Elevation (m)')
            plt.title('$Year$='+str(total_time_r))
            plt.legend(['Current topographic elevation', 'Initial topographic elevation', 'Current bedrock elevation', 'Initial bedrock elevation'], loc = 'upper left', fontsize=6)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/xs_C/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            plt.close(fig) 

        # Export timeseries
        np.savetxt(Directory + 'TerrainSandbox/CSV/timesteps.csv', timesteps, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/times.csv', times, delimiter = ",")
        #
        np.savetxt(Directory + 'TerrainSandbox/CSV/z_means.csv', z_means, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/z_medians.csv', z_medians, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/z_2s.csv', z_2s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/z_98s.csv', z_98s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/z_maxs.csv', z_maxs, delimiter = ",")
        #
        np.savetxt(Directory + 'TerrainSandbox/CSV/dbedelevdt_means.csv', dbedelevdt_means, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dbedelevdt_medians.csv', dbedelevdt_medians, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dbedelevdt_2s.csv', dbedelevdt_2s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dbedelevdt_98s.csv', dbedelevdt_98s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dbedelevdt_maxs.csv', dbedelevdt_maxs, delimiter = ",")
        #
        np.savetxt(Directory + 'TerrainSandbox/CSV/dsoildepthdt_means.csv', dsoildepthdt_means, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dsoildepthdt_medians.csv', dsoildepthdt_medians, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dsoildepthdt_2s.csv', dsoildepthdt_2s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dsoildepthdt_98s.csv', dsoildepthdt_98s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dsoildepthdt_maxs.csv', dsoildepthdt_maxs, delimiter = ",")
        #
        np.savetxt(Directory + 'TerrainSandbox/CSV/dzdt_means.csv', dzdt_means, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dzdt_medians.csv', dzdt_medians, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dzdt_2s.csv', dzdt_2s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dzdt_98s.csv', dzdt_98s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/dzdt_maxs.csv', dzdt_maxs, delimiter = ",")
        #
        np.savetxt(Directory + 'TerrainSandbox/CSV/erosionrate_means.csv', erosionrate_means, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/erosionrate_medians.csv', erosionrate_medians, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/erosionrate_2s.csv', erosionrate_2s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/erosionrate_98s.csv', erosionrate_98s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/erosionrate_maxs.csv', erosionrate_maxs, delimiter = ",")
        #
        np.savetxt(Directory + 'TerrainSandbox/CSV/netdz_means.csv', netdz_means, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/netdz_medians.csv', netdz_medians, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/netdz_2s.csv', netdz_2s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/netdz_98s.csv', netdz_98s, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/netdz_maxs.csv', netdz_maxs, delimiter = ",")
        #
        np.savetxt(Directory + 'TerrainSandbox/CSV/sediment_flux_gauge.csv', sediment_flux_gauge, delimiter = ",")
        np.savetxt(Directory + 'TerrainSandbox/CSV/sediment_flux_outlet.csv', sediment_flux_outlet, delimiter = ",")
        
        # Reset Plot_Ticker    
        Plot_Ticker = 0
        
    # Export_DEM    
    if Export_DEM == True:
        if total_time_r == Export_DEM_Ticker:
            if path.exists(str(Directory)+'/TerrainSandbox') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox')
        if Export_DEM_Ticker >= Export_DEM_Interval - 0.0000000001:
            if path.exists(str(Directory)+'/TerrainSandbox/Export_DEM') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Export_DEM')
            write_esri_ascii(str(Directory)+'/TerrainSandbox/Export_DEM/'+str(total_time_r)+'.asc', mg, names='topographic__elevation', clobber = True)
            Export_DEM_Ticker = 0
            
print('')
print('Complete!')

#%%