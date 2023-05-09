#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from mat4py import loadmat
import netCDF4
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import mean
import netCDF4 as nc
import nc_time_axis
import xarray as xr
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import scipy.io as spio
from matplotlib.pyplot import fill_betweenx
import os
from scipy import interpolate
import matplotlib.colors as colors
import string
import warnings
import seaborn as sns
from matplotlib.legend import Legend


# In[2]:


#creating data arrays for the observational data
dm= spio.loadmat('/Users/nuhahameed/Documents/Data for MSci/ITPAJX_Robert .mat')
mo=dm['AJX'][0,0]['mo'][0]
nl=np.zeros(12)
TA=np.zeros([12,500])
SA=np.zeros([12,500])
for i,m in enumerate(mo):
    TA[m-1][np.where(~np.isnan(dm['AJX'][0,0]['T'][i,:]))]+=dm['AJX'][0,0]['T'][i,:][np.where(~np.isnan(dm['AJX'][0,0]['T'][i,:]))]
    SA[m-1][np.where(~np.isnan(dm['AJX'][0,0]['S'][i,:]))]+=dm['AJX'][0,0]['S'][i,:][np.where(~np.isnan(dm['AJX'][0,0]['S'][i,:]))]
    nl[m-1]+=1
TA/=nl[:,np.newaxis]
SA/=nl[:,np.newaxis]

mo=dm['ITP'][0,0]['mo'][0]
nl=np.zeros(12)
TI=np.zeros([12,500])
SI=np.zeros([12,500])
for i,m in enumerate(mo):
    TI[m-1][np.where(~np.isnan(dm['ITP'][0,0]['T'][i,:]))]+=dm['ITP'][0,0]['T'][i,:][np.where(~np.isnan(dm['ITP'][0,0]['T'][i,:]))]
    SI[m-1][np.where(~np.isnan(dm['ITP'][0,0]['S'][i,:]))]+=dm['ITP'][0,0]['S'][i,:][np.where(~np.isnan(dm['ITP'][0,0]['S'][i,:]))]
#    TI[m-1]+=dm['ITP'][0,0]['T'][i,:]
#    SI[m-1]+=dm['ITP'][0,0]['S'][i,:]
    nl[m-1]+=1
TI/=nl[:,np.newaxis]
SI/=nl[:,np.newaxis]


# In[3]:


depth_res = 1  # depth resolution in meters
depth_values = np.arange(500) * depth_res  # create array of depth values

AJX_avg_salinity = np.mean(SA, axis=0)  # compute average salinity across all months
ITP_avg_salinity = np.nanmean(SI, axis=0) #compute average salinity across all months


# In[4]:


depth_range = (depth_values >= 0) & (depth_values <= 150)

#AIDJEX
AJX_salinity_depth_range = AJX_avg_salinity[depth_range]  #salinity profile up to 150 meters
AJX_Phi = np.sum(AJX_salinity_depth_range - 33) 

#ITP
ITP_salinity_depth_range = ITP_avg_salinity[depth_range]  #salinity profile up to 150 meters
ITP_Phi = np.sum(ITP_salinity_depth_range - 33)  

print("Phi for AIDJEX:", AJX_Phi)
print("Phi for ITP:", ITP_Phi)

# Calculate the ratio of Phi values for AIDJEX and ITP
Phi_ratio = ITP_Phi / AJX_Phi

print("Phi ratio (AIDJEX to ITP):", Phi_ratio)




# In[5]:


import matplotlib.pyplot as plt

# Extract salinity at depth 0 for each month from May to December
SA_depth0 = SA[4:12, 0]  # May is at index 4 (0-based), depth 0 is at index 0
SI_depth0 = SI[4:12, 0]

# Create an array for the months (May to December)
months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot the data
plt.plot(months, SA_depth0, marker='o', label='AJX (SA)')
plt.plot(months, SI_depth0, marker='o', label='ITP (SI)')

# Customize the plot
plt.xlabel('Months')
plt.ylabel('Salinity at Depth 0')
plt.title('Salinity at Depth 0 (May to December)')
plt.legend()

# Show the plot
plt.show()


# In[6]:


ratio = ITP_Phi / AJX_Phi

depth_range = (depth_values >= 0) & (depth_values <= 150)
depth_range_values = depth_values[depth_range]

# rescaled AJX profile
rescaled_AJX_salinity_depth_range = (AJX_salinity_depth_range - 33) * ratio + 33

# Plot 
plt.plot(AJX_salinity_depth_range, depth_range_values, label='Original AIDJEX', color='red', linewidth=4)
plt.plot(rescaled_AJX_salinity_depth_range, depth_range_values, label='Rescaled AIDJEX', color='blue', linewidth=4)
plt.plot(ITP_salinity_depth_range, depth_range_values, label='ITP', color='orange', linewidth=4)

plt.gca().invert_yaxis() 
plt.xlabel('Salinity (g/kg)', fontsize =14)
plt.ylabel('Depth (m)', fontsize =14)
plt.title('Rescaled AIDJEX Salinity Profile', fontsize=15)
plt.legend()
plt.savefig('rescaled AJX.png')
plt.show()


# In[8]:


# Calculate derivatives for both datasets
dSI_dz_AJX = np.gradient(AJX_avg_salinity, depth_values)
dSI_dz_ITP = np.gradient(ITP_avg_salinity, depth_values)
# Find the maximum gradient and its corresponding depth for both datasets
max_gradient_AJX = np.max(dSI_dz_AJX)
max_gradient_depth_AJX = depth_values[np.argmax(dSI_dz_AJX)]

max_gradient_ITP = np.max(dSI_dz_ITP)
max_gradient_depth_ITP = depth_values[np.argmax(dSI_dz_ITP)]

# Create a dictionary to store the dataset names, maximum gradients, and corresponding depths
data = {'Dataset': ['AIDJEX', 'ITP'],
        'Max Gradient': [max_gradient_AJX, max_gradient_ITP],
        'Depth of Max Gradient': [max_gradient_depth_AJX, max_gradient_depth_ITP]}

# Convert the dictionary to a pandas DataFrame
df_observations = pd.DataFrame(data)
df_observations.to_excel('Observations.xlsx', index=False)


# Display the DataFrame
print(df_observations)


# In[19]:


# Calculate derivatives for both datasets
dSI_dz_AJX = np.gradient(AJX_avg_salinity, depth_values)
dSI_dz_ITP = np.gradient(ITP_avg_salinity, depth_values)
dSI_rescaled_AJX_salinity_depth_range = np.gradient(rescaled_AJX_salinity_depth_range, depth_range_values)

# Plot the derivatives
plt.plot(dSI_dz_AJX, depth_values, label='1975 AIDJEX', color='red', linewidth=4)
plt.plot(dSI_dz_ITP, depth_values, label='2006-2012 ITP', color='orange', linewidth=4)
plt.plot(dSI_rescaled_AJX_salinity_depth_range, depth_range_values, label='Rescaled AIDJEX', color='blue', linewidth=4)

# Set y-axis limits and invert the y-axis
plt.ylim(150, 0)

# Add labels, a title, and a legend to the plot
plt.xlabel('Salinity Gradient (g/kg/m)', fontsize=14)
plt.ylabel('Depth (m)', fontsize=14)
plt.title('Vertical Salinity Gradient', fontsize=15)
plt.legend()
plt.savefig('vertical salinity gradiet observations')
plt.show()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming the 'SA' and 'SI' arrays have already been defined and filled with data

depth_res = 1  # depth resolution in meters
depth_values = np.arange(500) * depth_res  # create array of depth values

max_gradient_depths_AJX = []
max_gradient_depths_ITP = []

for month_idx in range(4, 12):  # Loop over months from May (index 4) to December (index 11)
    # Calculate gradients for the salinity profiles of both datasets
    dSA_dz_AJX = np.gradient(SA[month_idx, :], depth_values)
    dSI_dz_ITP = np.gradient(SI[month_idx, :], depth_values)
    
    # Find the depth of the maximum gradient for both datasets
    max_gradient_depth_AJX = depth_values[np.argmax(dSA_dz_AJX)]
    max_gradient_depth_ITP = depth_values[np.argmax(dSI_dz_ITP)]
    
    max_gradient_depths_AJX.append(max_gradient_depth_AJX)
    max_gradient_depths_ITP.append(max_gradient_depth_ITP)

# Create an array for the months (May to December)
months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot the depth of the maximum gradient for both datasets
plt.plot(months, max_gradient_depths_AJX, label='AJX (SA)')
plt.plot(months, max_gradient_depths_ITP, label='ITP (SI)')

# Customize the plot
plt.xlabel('Months')
plt.ylabel('Depth of Maximum Gradient')
plt.title('Depth of Maximum Gradient for Salinity Profiles (May to December)')
plt.legend()

# Show the plot
plt.show()


# In[10]:


warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
input_folder ='/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

fig, axs = plt.subplots(8, 4, layout='tight', figsize=(5 * 5, 5 * 8), sharex=True)
axs.reshape(-4)
axs[-1, -1].remove()
axs[-1, -2].remove()  

def int_to_roman(integer):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    i = 0
    while integer > 0:
        for _ in range(integer // val[i]):
            roman_num += syb[i]
            integer -= val[i]
        i += 1
    return roman_num

def figureone(file, i):
    ds = xr.open_dataset(input_folder+file) #opens the dataset
    if 'CESM' in file: 
        ds['lev'] = ds['lev']/100
        
    model_name = file.split('so.')[-1].split('.')[0]

    Year = ['1970', '1980', '1990', '2000', '2010']
    
    for j, year in enumerate(Year):
        ds_mean = ds.so.isel(time=np.arange((int(year)-1850)*12, (int(year)+1-1850)*12,1)).mean(dim='time')
        y = ds_mean.mean(dim=['longitude', 'latitude']) #works out the average within the longitudanal range
        
        ds_std = ds.so.isel(time=np.arange((int(year)-1850)*12, (int(year)+1-1850)*12,1)).std(dim='time')
        ds_std_latlon = ds_std.std(dim = ['longitude', 'latitude'])

        y_upper = y + ds_std_latlon
        y_lower = y - ds_std_latlon
        
        axs.reshape(-1)[i].plot(y, ds['lev'], label=year, color=plt.cm.viridis(j/len(Year)), linewidth = 4)
        
        for year in range(1970, 2011):
            ds_year = ds.so.isel(time=np.arange((year-1850)*12, (year+1-1850)*12, 1)).mean(dim='time')
            y_year = ds_year.mean(dim=['longitude', 'latitude'])
            axs.reshape(-1)[i].plot(y_year, ds['lev'], color='gray', alpha=0.3, linewidth=0.3)

        axs.reshape(-1)[i].fill_betweenx(ds['lev'], y_upper, y_lower, color=plt.cm.viridis(j/len(Year)), alpha=0.2)
        
        axs.reshape(-1)[i].set_xlabel('Salinity (g/kg)', fontsize = 20)
        axs.reshape(-1)[i].set_ylabel('Depth (m)', fontsize = 20)
        axs.reshape(-1)[i].set_ylim(0,150)
        axs.reshape(-1)[i].set_xlim(27,35)
        axs.reshape(-1)[i].tick_params(axis='x', labelsize=18)
        axs.reshape(-1)[i].tick_params(axis='y', labelsize=18)
        axs.reshape(-1)[i].xaxis.set_tick_params(which='both', labelbottom=True)

    axs.reshape(-1)[i].invert_yaxis()
    axs.reshape(-1)[i].set_title(model_name, fontsize = 25)
    axs.reshape(-1)[i].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth = 4)
    axs.reshape(-1)[i].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth =4)
    
    
    if i < 26:
        subplot_label = string.ascii_lowercase[i]
    else:
        subplot_label = int_to_roman((i - 25))

    axs.reshape(-1)[i].text(0.02, 1.1, subplot_label, transform=axs.reshape(-1)[i].transAxes, fontsize=16, fontweight='bold', va='top', ha='left')


i=0
for file in myfiles:    
    figureone(file, i)
    i=i+1 
    
Year = ['1970', '1980', '1990', '2000', '2010']
year_colors = [plt.cm.viridis(j/len(Year)) for j in range(len(Year))]
year_handles = [plt.Line2D([0], [0], color=color, lw=4, label=year) for year, color in zip(Year, year_colors)]
aidjex_handle = plt.Line2D([0], [0], color='red', lw=4, label='1975 AIDJEX')
itp_handle = plt.Line2D([0], [0], color='orange', lw=4, label='2006-2012 ITP')
legend_handles = year_handles + [aidjex_handle, itp_handle]

# Customize the position of the legend by changing x and y
x, y = 0.713, 0.12
fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], fontsize=30, bbox_to_anchor=(x, y), ncol=1, borderaxespad=0)
plt.savefig('Salinity_profiles_09_04.png')


# In[11]:


input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)
        
def calculate_phi(salinity, depth_range, s0, S=None):
    salinity_depth_range = salinity[depth_range]
    if S is not None:
        salinity_depth_range[0] = S
    phi = np.trapz(s0 - salinity_depth_range, depth_range)
    return phi


def phi_calculation(file):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    phi_values = {}
    for year in ['1970', '2010']:
        ds_mean = ds.so.isel(time=np.arange((int(year) - 1850) * 12, (int(year) + 1 - 1850) * 12, 1)).mean(dim='time')
        y = ds_mean.mean(dim=['longitude', 'latitude'])
        
        # Find the nearest depth to 0 and make the max 150
        depth_start = np.argmin(np.abs(ds['lev'].values - 0))
        depth_end = np.argmin(np.abs(ds['lev'].values - 150))

        # Select the depth range
        depth_range_values = np.arange(depth_start, depth_end + 1)
        
        # Calculate s0 as the salinity at depth 150
        s0 = y.sel(lev=150, method='nearest').values
        
        # Find the nearest depth to 0 and use it to get S
        depth_nearest_0 = np.argmin(np.abs(ds['lev'].values - 0))
        S = y.sel(lev=ds['lev'].values[depth_nearest_0], method='nearest').values
        
        # Calculate Phi using the updated calculate_phi function
        salinity_profile = y.values[depth_range_values]
        phi = calculate_phi(salinity_profile, depth_range_values, s0, S)
        phi_values[year] = phi

    return phi_values


for file in myfiles:
    model_name = file.split('so.')[-1].split('.')[0]
    phi_values = phi_calculation(file)

results = pd.DataFrame(columns=['Model', 'Phi_1970', 'Phi_2010', 'Phi_ratio'])

for file in myfiles:
    model_name = file.split('so.')[-1].split('.')[0]
    phi_values = phi_calculation(file)

    results = results.append({
        'Model': model_name,
        'Phi_1970': phi_values['1970'],
        'Phi_2010': phi_values['2010'],
        'Phi_ratio': phi_values['2010'] / phi_values['1970']
    }, ignore_index=True)


results.to_excel('model_phi.xlsx', index=False)
print(results)


# In[12]:


input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

def calculate_phi(salinity, depth_range):
    salinity_depth_range = salinity[depth_range]
    Phi = np.trapz(salinity_depth_range - 33)
    return Phi

def phi_calculation(file):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    phi_values = {}
    for year in ['1970', '2010']:
        ds_mean = ds.so.isel(time=np.arange((int(year) - 1850) * 12, (int(year) + 1 - 1850) * 12, 1)).mean(dim='time')
        y = ds_mean.mean(dim=['longitude', 'latitude'])
        
        # Find the nearest depth to 0 and make the max 150
        depth_start = np.argmin(np.abs(ds['lev'].values - 0))
        depth_end = np.argmin(np.abs(ds['lev'].values - 150))

        # Select the depth range
        depth_range_values = np.arange(depth_start, depth_end + 1)
        
        salinity_profile = y.values[depth_range_values]
        phi = calculate_phi(salinity_profile, depth_range_values)
        phi_values[year] = phi

    return phi_values

depth_range = np.arange(0, 151)

for file in myfiles:
    model_name = file.split('so.')[-1].split('.')[0]
    phi_values = phi_calculation(file)

results = pd.DataFrame(columns=['Model', 'Phi_1970', 'Phi_2010', 'Phi_ratio'])

for file in myfiles:
    model_name = file.split('so.')[-1].split('.')[0]
    phi_values = phi_calculation(file)

    results = results.append({
        'Model': model_name,
        'Phi_1970': phi_values['1970'],
        'Phi_2010': phi_values['2010'],
        'Phi_ratio': phi_values['2010'] / phi_values['1970']
    }, ignore_index=True)


results.to_excel('model_phi.xlsx', index=False)
print(results)


# In[28]:


input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

colors = sns.color_palette("husl", len(myfiles))
linestyles = ['-', '--', '-.', ':'] * (len(myfiles) // 4 + 1)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(29, 12), sharex=True)
plt.subplots_adjust(right=0.7, hspace=0.3)

def plot_salinity_profile(ax, file, color, linestyle, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    #start_year = 2000
    #end_year = 2010
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])
    
    ax.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle=linestyle)

for file, color, linestyle in zip(myfiles, colors, linestyles):
    plot_salinity_profile(axs[0], file, color, linestyle, 1970, 1980)
    plot_salinity_profile(axs[1], file, color, linestyle, 2000, 2010)

axs[0].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth = 4)
axs[1].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth=4)

for i, ax in enumerate(axs):
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Salinity (g/kg)', fontsize=14)
    ax.set_ylim(0, 150)
    ax.set_xlim(27, 35)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.invert_yaxis()
    
    # Add subplot labels
    label = chr(97 + i)  # Convert index to ASCII character ('a', 'b', etc.)
    ax.text(-0.1, 1, label, fontsize=19, fontweight='bold', transform=ax.transAxes)
    
axs[0].set_title('1970-1980 Salinity Profiles', fontsize=25)
axs[1].set_title('2000-2010 Salinity Profiles', fontsize=25)

# Get the handles and labels from the first subplot
handles, labels = axs[0].get_legend_handles_labels()

# Add the handle and label for the ITP line from the second subplot
handles_itp, labels_itp = axs[1].get_legend_handles_labels()
handles += handles_itp[-1:]
labels += labels_itp[-1:]

# Create a shared legend for all subplots
fig.legend(handles, labels, fontsize=15, bbox_to_anchor=(0.68, -0.1, 0.15, 1), ncol=1, borderaxespad=0)
    

plt.savefig('multi_model_Salinity_Profiles_13_04_.png', bbox_inches='tight')


# In[16]:


input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

colors = sns.color_palette("husl", len(myfiles))
linestyles = ['-', '--', '-.', ':'] * (len(myfiles) // 4 + 1)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(29, 12), sharex=True)
plt.subplots_adjust(right=0.7, hspace=0.3)

def plot_salinity_profile(ax, file, color, linestyle, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    #start_year = 2000
    #end_year = 2010
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])
    
    axs.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle=linestyle)

for file, color, linestyle in zip(myfiles, colors, linestyles):
    plot_salinity_profile(axs[0], file, color, linestyle, 1970, 1980)
    plot_salinity_profile(axs[1], file, color, linestyle, 2000, 2010)

axs[0].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth = 4)
axs[1].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth=4)

for i, ax in enumerate(axs):
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Salinity (g/kg)', fontsize=14)
    ax.set_ylim(0, 150)
    ax.set_xlim(27, 35)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.invert_yaxis()
    
    # Add subplot labels
    label = chr(97 + i)  # Convert index to ASCII character ('a', 'b', etc.)
    ax.text(-0.1, 1, label, fontsize=19, fontweight='bold', transform=ax.transAxes)
    
axs[0].set_title('1970-1980 Salinity Profiles', fontsize=20)
axs[1].set_title('2000-2010 Salinity Profiles', fontsize=20)

# Get the handles and labels from the first subplot
handles, labels = axs[0].get_legend_handles_labels()

# Add the handle and label for the ITP line from the second subplot
handles_itp, labels_itp = axs[1].get_legend_handles_labels()
handles += handles_itp[-1:]
labels += labels_itp[-1:]

# Create a shared legend for all subplots
fig.legend(handles, labels, fontsize=15, bbox_to_anchor=(0.68, -0.1, 0.15, 1), ncol=1, borderaxespad=0)

excel_file_path = '/Volumes/Seagate Hub/ocean_components.xlsx'
model_info_df = pd.read_excel(excel_file_path, engine='openpyxl', sheet_name='Sheet1', header=None, names=['Model Name', 'Ocean Component'])


ocean_component_colors = {
    'MOM4-L40':'magenta',
    'MOM4':'red',
    'MOM6':'saddlebrown',
    'NEMO3.6': 'green',
    'Nemo3.4.1': 'lime',
    'NEMO-HadGEM3-GO6.0':'darkturquoise',
    'NEMO-LIM3.3.6':'darkkhaki',
    'GISS Ocean':'teal',
    'POP2': 'darkorange',
    'MPIOM1.6.3':'purple',
    'MRI.COM4.4':'lightskyblue',
    'MPAS-Ocean':'blue',
}

# Create a new column 'Color' in the model_info_df DataFrame based on the ocean components
model_info_df['Color'] = model_info_df['Ocean Component'].map(ocean_component_colors)

def plot_salinity_profile_ocean_component(ax, file, color, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])
    
    axs.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle='-')
    # Rest of the plot_salinity_profile function remains the same

for file in myfiles:
    model_name = file.split('so.')[-1].split('.')[0]
    color = model_info_df.loc[model_info_df['Model Name'] == model_name, 'Color'].values[0]

    plot_salinity_profile_ocean_component(axs[3], file, color, start_year=1970, end_year=1980)
    plot_salinity_profile_ocean_component(axs[4], file, color, start_year=2000, end_year=2010)
    
axs[2].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='black', linewidth = 4)
axs[3].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='black', linewidth=4)

    
for i, ax in enumerate(axs):
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Salinity (g/kg)', fontsize=14)
    ax.set_ylim(0, 150)
    ax.set_xlim(27, 35)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.invert_yaxis()
    
    
    # Add subplot labels
    label = chr(97 + i)  # Convert index to ASCII character ('a', 'b', etc.)
    ax.text(-0.1, 1, label, fontsize=19, fontweight='bold', transform=ax.transAxes)
    
axs[3].set_title('1970-1980 Salinity Profiles', fontsize=25)
axs[4].set_title('2000-2010 Salinity Profiles', fontsize=25)

unique_ocean_components = model_info_df['Ocean Component'].dropna().unique()
unique_colors = [ocean_component_colors[oc] for oc in unique_ocean_components]

# Create custom legend handles
legend_handles = [plt.Line2D([0], [0], color=color, lw=2, label=ocean_component) for ocean_component, color in zip(unique_ocean_components, unique_colors)]

# Create a shared legend for all subplots
fig.legend(legend_handles, unique_ocean_components, fontsize=15, bbox_to_anchor=(0.68, -0.1, 0.15, 1), ncol=1, borderaxespad=0)



# In[ ]:


# Define input folder path
input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'

# Create list of all files in folder
all_files = os.listdir(input_folder)

# Remove any hidden files from the list
all_files = [f for f in all_files if not f.startswith('.')]

def calc_mean_so(ds, year):
    #Year = ['1970', '1980', '1990', '2000', '2010']
    salinity = []
    #for j, year in enumerate(Year):
    ds_mean = ds.so.isel(time=np.arange((int(year)-1850)*12, (int(year)+1-1850)*12,1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude']) #works out the average within the longitudanal range
    salinity.append(y)
        #print(len(y))
    #else:
     #   print(f"No data found for {year}")
    return salinity


# In[14]:


# Loop over all files and calculate decadal average salinity data
all_salinity_depth_data = []
#all_salinity_lat_lon_data = []

all_y = []


for file in all_files:
    salinity_depth_data = calc_mean_so(xr.open_dataset(input_folder+file), '1970')
    all_salinity_depth_data.append(salinity_depth_data)
   # all_salinity_lat_lon_data.append(salinity_lat_lon_data)


#all_salinity_depth_data

new_list = []


def interp(array):
    x_out = np.arange(0, 150, 1)
    y_in = array.values
    x_in = array.lev.values
    f = interpolate.interp1d(x_in, y_in, fill_value='extrapolate') #option: change to fill_value='extrapolate'
    y_out = f(x_out)
    return y_out
    
    
for i in range(len(all_salinity_depth_data)):
    array = all_salinity_depth_data[i]
    new_array = interp(array[0])
    new_list.append(new_array)
    

mean_out_1970 = np.mean(new_list, axis=0)


# In[17]:


# Loop over all files and calculate decadal average salinity data
all_salinity_depth_data = []
#all_salinity_lat_lon_data = []

all_y = []


for file in all_files:
    salinity_depth_data = calc_mean_so(xr.open_dataset(input_folder+file), '1980')
    all_salinity_depth_data.append(salinity_depth_data)
   # all_salinity_lat_lon_data.append(salinity_lat_lon_data)


#all_salinity_depth_data

new_list = []


def interp(array):
    x_out = np.arange(0, 150, 1)
    y_in = array.values
    x_in = array.lev.values
    f = interpolate.interp1d(x_in, y_in, fill_value='extrapolate') #option: change to fill_value='extrapolate'
    y_out = f(x_out)
    return y_out
    
    
for i in range(len(all_salinity_depth_data)):
    array = all_salinity_depth_data[i]
    new_array = interp(array[0])
    new_list.append(new_array)
    

mean_out_1980 = np.mean(new_list, axis=0)


# In[18]:


# Loop over all files and calculate decadal average salinity data
all_salinity_depth_data = []
#all_salinity_lat_lon_data = []

all_y = []


for file in all_files:
    salinity_depth_data = calc_mean_so(xr.open_dataset(input_folder+file), '1990')
    all_salinity_depth_data.append(salinity_depth_data)
   # all_salinity_lat_lon_data.append(salinity_lat_lon_data)


#all_salinity_depth_data

new_list = []


def interp(array):
    x_out = np.arange(0, 150, 1)
    y_in = array.values
    x_in = array.lev.values
    f = interpolate.interp1d(x_in, y_in, fill_value='extrapolate') #option: change to fill_value='extrapolate'
    y_out = f(x_out)
    return y_out
    
    
for i in range(len(all_salinity_depth_data)):
    array = all_salinity_depth_data[i]
    new_array = interp(array[0])
    new_list.append(new_array)
    

mean_out_1990 = np.mean(new_list, axis=0)


# In[19]:


# Loop over all files and calculate decadal average salinity data
all_salinity_depth_data = []
#all_salinity_lat_lon_data = []

all_y = []


for file in all_files:
    salinity_depth_data = calc_mean_so(xr.open_dataset(input_folder+file), '2000')
    all_salinity_depth_data.append(salinity_depth_data)
   # all_salinity_lat_lon_data.append(salinity_lat_lon_data)


#all_salinity_depth_data

new_list = []


def interp(array):
    x_out = np.arange(0, 150, 1)
    y_in = array.values
    x_in = array.lev.values
    f = interpolate.interp1d(x_in, y_in, fill_value='extrapolate') #option: change to fill_value='extrapolate'
    y_out = f(x_out)
    return y_out
    
    
for i in range(len(all_salinity_depth_data)):
    array = all_salinity_depth_data[i]
    new_array = interp(array[0])
    new_list.append(new_array)
    

mean_out_2000 = np.mean(new_list, axis=0)


# In[20]:


# Loop over all files and calculate decadal average salinity data
all_salinity_depth_data = []
#all_salinity_lat_lon_data = []

all_y = []


for file in all_files:
    salinity_depth_data = calc_mean_so(xr.open_dataset(input_folder+file), '2010')
    all_salinity_depth_data.append(salinity_depth_data)
   # all_salinity_lat_lon_data.append(salinity_lat_lon_data)


#all_salinity_depth_data

new_list = []


def interp(array):
    x_out = np.arange(0, 150, 1)
    y_in = array.values
    x_in = array.lev.values
    f = interpolate.interp1d(x_in, y_in, fill_value='extrapolate') #option: change to fill_value='extrapolate'
    y_out = f(x_out)
    return y_out
    
    
for i in range(len(all_salinity_depth_data)):
    array = all_salinity_depth_data[i]
    new_array = interp(array[0])
    new_list.append(new_array)
    

mean_out_2010 = np.mean(new_list, axis=0)


# In[21]:


def plot_salinity_depth(years, salinity_depth_data):
    for i, data in enumerate(salinity_depth_data):
        Year = ['1970', '1980', '1990', '2000', '2010']
        plt.plot(data, np.arange(0, 150, 1), label=years[i], color=plt.cm.viridis(i/len(Year)), linewidth = 4)
    plt.ylim(150, 0)
    plt.xlim(27,35)
    plt.xlabel('Salinity (g/kg)', fontsize = 15)
    plt.ylabel('Depth (m)', fontsize = 15)
    plt.title(f"Multi Model Mean", fontsize = 20)
    
    plt.plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth = 4)
    plt.plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth = 4)
    plt.legend()
    plt.savefig('Multi Model Mean.png')
    plt.show()

# Loop over all files and calculate decadal average salinity data
all_salinity_depth_data = {}
for year in ['1970', '1980', '1990', '2000', '2010']:
    salinity_depth_data = []
    for file in all_files:
        salinity_depth_data.append(calc_mean_so(xr.open_dataset(input_folder+file), year))
    all_salinity_depth_data[year] = salinity_depth_data

# Interpolate the data
interpolated_salinity_data = {}
for year, salinity_depth_data in all_salinity_depth_data.items():
    interpolated_salinity_data[year] = [interp(array[0]) for array in salinity_depth_data]

# Calculate the mean salinity data for each decade
mean_salinity_data = {}
for year, interpolated_salinity in interpolated_salinity_data.items():
    mean_salinity_data[year] = np.mean(interpolated_salinity, axis=0)

# Plot the mean salinity data for all decades
plot_salinity_depth(['1970', '1980', '1990', '2000', '2010'], list(mean_salinity_data.values()))



# In[ ]:


# Define a function to calculate maximum salinity gradient and depth for a given model and time period
def max_gradient_depth(file, year='2010', max_depth=150):
    ds = xr.open_dataset(os.path.join(input_folder, file))
    if 'CESM' in file: 
        ds['lev'] = ds['lev']/100
    model_name = file.split('so.')[-1].split('.')[0]

    # Extract salinity data for the specified year
    ds_mean = ds.so.sel(lev=slice(0, max_depth)).isel(time=np.arange((int(year)-1850)*12, (int(year)+1-1850)*12,1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])

    # Calculate salinity gradient (derivative)
    d_values = ds.lev.sel(lev=slice(0, max_depth)).values
    ds_derivative = np.gradient(y, d_values, axis=0)

    # Find maximum salinity gradient and depth
    max_gradient = ds_derivative[:-2].max()
    depth_max_gradient = d_values[ds_derivative[:-2].argmax()]

    # Extract salinity value at depth 0 for the specified year
    salinity_at_depth_0 = ds['so'].sel(time='2010', lev=ds['lev'].sel(method='nearest', lev=0)).mean().values

    return {'Model': model_name, 'Max Gradient': max_gradient, 'Mixed layer depth': depth_max_gradient, 'Surface Salinity': salinity_at_depth_0}

# Create an empty list to store results
results = []

# Loop through each file and calculate maximum salinity gradient and depth for 2010
for file in myfiles:
    result = max_gradient_depth(file)
    results.append(result)

# Convert results to a pandas DataFrame
df = pd.DataFrame(results)

# Sort results by maximum gradient in descending order
#df = df.sort_values('Max Gradient', ascending=False)

# Print table
print(df[['Model', 'Surface Salinity', 'Max Gradient', 'Mixed layer depth']])


# In[ ]:


surface_salinity_1970 = [data[0] for data in interpolated_salinity_data['1970']]  # Salinity data at depth=0m for 1975
surface_salinity_2010 = [data[0] for data in interpolated_salinity_data['2010']]  # Salinity data at depth=0m for 2010
surface_salinity_diff = np.array(surface_salinity_1970) - np.array(surface_salinity_2010)

surface_salinity_diff[0]


# In[4]:


input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

# Remove any filenames that contain '._so'
myfiles = [f for f in myfiles if '._so' not in f]

# Create a list to store the differences in average salinity at depth 0
salinity_diffs = []

# Iterate over each file
for file in myfiles:
    ds = xr.open_dataset(input_folder+file)
    
    # Extract the model name from the filename
    model_name = file.split('so.')[-1].split('.')[0]
    
    # Calculate the mean salinity at depth 0 for each year
    salinity_1970 = ds['so'].sel(time='1970', lev=ds['lev'].sel(method='nearest', lev=0)).mean().values
    salinity_2010 = ds['so'].sel(time='2010', lev=ds['lev'].sel(method='nearest', lev=0)).mean().values
    
    # Calculate the difference in average salinity at depth 0 between 1970 and 2010
    salinity_diff = salinity_2010 - salinity_1970
    
    # Add the model name and salinity difference to the list
    salinity_diffs.append([model_name, salinity_diff])
    
# Create a pandas DataFrame from the list
df = pd.DataFrame(salinity_diffs, columns=['Model name', 'Salinity difference'])
df = df.sort_values('Salinity difference', ascending=False)


df.to_excel('So_differences.xlsx', index=False)

print(df)


# In[13]:


# Plot a histogram of the salinity difference values
plt.hist(df['Salinity difference'], bins=10)
plt.title('Surface Salinity Difference', fontsize=15)
plt.xlabel('Difference in surface salinity between 2010 and 1970 (g/kg)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('hist_sal_diff')


# In[28]:


obs_so_diff = ITP_avg_salinity[0:1] - AJX_avg_salinity[0:1] 
#obs_so_diff[0:1]
obs_so_diff


# In[29]:


excel_file_path = '/Volumes/Seagate Hub/ocean_components.xlsx'
model_info_df = pd.read_excel(excel_file_path, engine='openpyxl', sheet_name='Sheet1', header=None, names=['Model Name', 'Ocean Component'])


# Display the DataFrame
print(model_info_df)                    


# In[11]:


# Read the Excel sheet into a pandas DataFrame
excel_file_path = '/Volumes/Seagate Hub/ocean_components.xlsx'
model_info_df = pd.read_excel(excel_file_path, engine='openpyxl', sheet_name='Sheet1', header=None, names=['Model Name', 'Ocean Component'])


ocean_component_colors = {
    'MOM4-L40':'magenta',
    'MOM4':'navy',
    'MOM6':'saddlebrown',
    'NEMO3.6': 'green',
    'Nemo3.4.1': 'lime',
    'NEMO-HadGEM3-GO6.0':'darkturquoise',
    'NEMO-LIM3.3.6':'darkkhaki',
    'GISS Ocean':'teal',
    'POP2': 'lightcoral',
    'MPIOM1.6.3':'purple',
    'MRI.COM4.4':'lightskyblue',
    'MPAS-Ocean':'slategrey',
    'EC-Earth NEMO3.6': 'blue',
}

# Create a new column 'Color' in the model_info_df DataFrame based on the ocean components
model_info_df['Color'] = model_info_df['Ocean Component'].map(ocean_component_colors)

# Rest of your code

input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(29, 12), sharex=True)
plt.subplots_adjust(right=0.7, hspace=0.3)

def plot_salinity_profile_OC(ax, file, color, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])
    
    ax.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle='-')
    # Rest of the plot_salinity_profile function remains the same

for file in myfiles:
    model_name = file.split('so.')[-1].split('.')[0]
    color = model_info_df.loc[model_info_df['Model Name'] == model_name, 'Color'].values[0]

    plot_salinity_profile_OC(axs[0], file, color, start_year=1970, end_year=1980)
    plot_salinity_profile_OC(axs[1], file, color, start_year=2000, end_year=2010)
    
axs[0].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth = 5)
axs[1].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth=5)

    
for i, ax in enumerate(axs):
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Salinity (g/kg)', fontsize=14)
    ax.set_ylim(0, 150)
    ax.set_xlim(27, 35)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.invert_yaxis()
    
    
    label = chr(99 + i)  # Convert index to ASCII character ('c', 'd', etc.)
    ax.text(-0.1, 1, label, fontsize=19, fontweight='bold', transform=ax.transAxes)

# ... rest of your code above

unique_ocean_components = model_info_df['Ocean Component'].dropna().unique()
unique_colors = [ocean_component_colors[oc] for oc in unique_ocean_components]

# Create custom legend handles
legend_handles = [plt.Line2D([0], [0], color=color, lw=2, label=ocean_component) for ocean_component, color in zip(unique_ocean_components, unique_colors)]

# Add custom legend handles for '1975 AIDJEX' and '2006-2012 ITP'
aidjex_handle = plt.Line2D([0], [0], color='red', lw=5, label='1975 AIDJEX')
itp_handle = plt.Line2D([0], [0], color='orange', lw=5, label='2006-2012 ITP')
legend_handles.extend([aidjex_handle, itp_handle])

# Create a shared legend for all subplots
fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], fontsize=15, bbox_to_anchor=(0.68, -0.1, 0.15, 1), ncol=1, borderaxespad=0)

plt.savefig('ocean_component_plot.png', bbox_inches='tight')


# In[12]:


# Read the Excel sheet into a pandas DataFrame
excel_file_path = '/Volumes/Seagate Hub/ocean_components.xlsx'
model_info_df = pd.read_excel(excel_file_path, engine='openpyxl', sheet_name='Sheet1', header=None, names=['Model Name', 'Ocean Component'])

input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)


for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(29, 12), sharex=True)
plt.subplots_adjust(right=0.7, hspace=0.3)


unique_ocean_components = model_info_df['Ocean Component'].dropna().unique()
color_palette = sns.color_palette('tab10', n_colors=len(unique_ocean_components))
ocean_component_colors = dict(zip(unique_ocean_components, color_palette))

# Create a new column 'Color' in the model_info_df DataFrame based on the ocean components
model_info_df['Color'] = model_info_df['Ocean Component'].map(ocean_component_colors)

# Rest of your code


def plot_salinity_profile_OC(ax, file, color, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])
    
    ax.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle='-')
    # Rest of the plot_salinity_profile function remains the same

for file in myfiles:
    model_name = file.split('so.')[-1].split('.')[0]
    color = model_info_df.loc[model_info_df['Model Name'] == model_name, 'Color'].values[0]

    plot_salinity_profile_OC(axs[0], file, color, start_year=1970, end_year=1980)
    plot_salinity_profile_OC(axs[1], file, color, start_year=2000, end_year=2010)
    
axs[0].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth = 5)
axs[1].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth=5)

    
for i, ax in enumerate(axs):
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Salinity (g/kg)', fontsize=14)
    ax.set_ylim(0, 150)
    ax.set_xlim(27, 35)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.invert_yaxis()
    
    
    label = chr(99 + i)  # Convert index to ASCII character ('c', 'd', etc.)
    ax.text(-0.1, 1, label, fontsize=19, fontweight='bold', transform=ax.transAxes)

# ... rest of your code above

unique_ocean_components = model_info_df['Ocean Component'].dropna().unique()
unique_colors = [ocean_component_colors[oc] for oc in unique_ocean_components]

# Create custom legend handles
legend_handles = [plt.Line2D([0], [0], color=ocean_component_colors[oc], lw=2, label=oc) for oc in unique_ocean_components]
aidjex_handle = plt.Line2D([0], [0], color='red', lw=5, label='1975 AIDJEX')
itp_handle = plt.Line2D([0], [0], color='orange', lw=5, label='2006-2012 ITP')
legend_handles.extend([aidjex_handle, itp_handle])

# Create a shared legend for all subplots
fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], fontsize=15, bbox_to_anchor=(0.68, -0.1, 0.15, 1), ncol=1, borderaxespad=0)

plt.savefig('ocean_component_plot.png', bbox_inches='tight')


# In[25]:


import os
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

colors = sns.color_palette("husl", len(myfiles))
linestyles = ['-', '--', '-.', ':'] * (len(myfiles) // 4 + 1)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(25, 20), sharex=True)
plt.subplots_adjust(right=0.7, hspace=0.3)

def plot_salinity_profile(ax, file, color, linestyle, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])

    ax.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle=linestyle)

for file, color, linestyle in zip(myfiles, colors, linestyles):
    plot_salinity_profile(axs[0, 0], file, color, linestyle, 1970, 1980)
    plot_salinity_profile(axs[0, 1], file, color, linestyle, 2000, 2010)

axs[0, 0].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth=4)
axs[0, 1].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth=4)

for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        ax.set_ylabel('Depth (m)', fontsize=14)
        ax.set_xlabel('Salinity (g/kg)', fontsize=14)
        ax.set_ylim(0, 150)
        ax.set_xlim(27, 35)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.invert_yaxis()

        # Add subplot labels
        label = chr(97 + i * 2 + j)  # Convert index to ASCII character ('a', 'b', etc.)
        ax.text(-0.1, 1, label, fontsize=19, fontweight='bold', transform=ax.transAxes)

axs[0, 0].set_title('1970-1980 Salinity Profiles', fontsize=20)
axs[0, 1].set_title('2000-2010 Salinity Profiles', fontsize=20)

def plot_salinity_profile_ocean_component(ax, file, color, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])

    ax.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle='-')

for file in myfiles:
    model_name = file.split('so.')[-1].split('.')[0]
    color = model_info_df.loc[model_info_df['Model Name'] == model_name, 'Color'].values[0]

    plot_salinity_profile_ocean_component(axs[1, 0], file, color, start_year=1970, end_year=1980)
    plot_salinity_profile_ocean_component(axs[1, 1], file, color, start_year=2000, end_year=2010)

axs[1, 0].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='black', linewidth=4)
axs[1, 1].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='black', linewidth=4)



# Get the handles and labels from the first subplot
handles, labels = axs[0, 0].get_legend_handles_labels()

# Add the handle and label for the ITP line from the second subplot
handles_itp, labels_itp = axs


# In[10]:


import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = [f for f in os.listdir(input_folder) if not f.startswith('._') and 'so' in f]

fig, axs = plt.subplots(8, 4, layout='tight', figsize=(5 * 5, 5 * 8), sharex=True)
axs[-1, -1].remove()
axs[-1, -2].remove()  
axs = axs.ravel()

def int_to_roman(integer):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    i = 0
    while integer > 0:
        for _ in range(integer // val[i]):
            roman_num += syb[i]
            integer -= val[i]
        i += 1
    return roman_num

# Extract salinity at depth 0 for each month from May to December
SA_depth0 = SA[4:12, 0]  # May is at index 4 (0-based), depth 0 is at index 0
SI_depth0 = SI[4:12, 0]

# Create an array for the months (May to December)
months_obs = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def monthly_mean(file, i):
    print(file)
    ds = xr.open_dataset(input_folder + file)
    
    if 'CESM' in file: 
        ds['lev'] = ds['lev'] / 100
    
    model_name = file.split('so.')[-1].split('.')[0]

    ds_month = ds.resample(time="M").mean()
    ds_month_mean = ds_month.mean(dim=['longitude', 'latitude'])
    ds_month_mean_depthzero = ds_month_mean.isel(lev=0)

    months_to_select = range(5, 13)  # May to December
    decades = [(1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010)]

    for j, (start_year, end_year) in enumerate(decades):
        month_indices = [(int(year) - 1850) * 12 + month - 1 for year in range(start_year, end_year) for month in months_to_select]
        monthly_decadal_means = ds_month_mean_depthzero['so'].isel(time=month_indices).groupby('time.month').mean(dim='time')
        monthly_decadal_std = ds_month_mean_depthzero['so'].isel(time=month_indices).groupby('time.month').std(dim='time')

        y_upper = monthly_decadal_means + monthly_decadal_std
        y_lower = monthly_decadal_means - monthly_decadal_std

        axs[i].plot(months_to_select, monthly_decadal_means, label=f"{start_year}-{end_year}",  linewidth = 4, color=plt.cm.viridis(j/len(decades)))
        axs[i].fill_between(months_to_select, y_upper, y_lower, color=plt.cm.viridis(j/len(decades)), alpha=0.2)

    axs[i].set_xlabel('Month', fontsize= 20)
    axs[i].set_ylabel('Salinity (g/kg)', fontsize= 20)
    axs[i].set_title(model_name, fontsize = 25)
    axs[i].set_ylim(25,33)
    axs[i].plot(months_to_select, SA_depth0, label='AIDJEX', color = 'red', linewidth = 4)
    axs[i].plot(months_to_select, SI_depth0, label='ITP', color = 'orange', linewidth =4)
    if i == len(axs) - 3:
        axs[i].legend(loc='lower left')

    axs[i].tick_params(axis='y', labelsize=18)
    axs[i].tick_params(axis='x', labelsize=18)
    axs[i].xaxis.set_tick_params(which='both', labelbottom=True)
    

    if i < 26:
        subplot_label = string.ascii_lowercase[i]
    else:
        subplot_label = int_to_roman((i - 25))

    axs.reshape(-1)[i].text(0.02, 1.1, subplot_label, transform=axs.reshape(-1)[i].transAxes, fontsize=16, fontweight='bold', va='top', ha='left')


i = 0
for file in myfiles:    
    monthly_mean(file, i)
    i += 1

plt.savefig('figure2.png')

