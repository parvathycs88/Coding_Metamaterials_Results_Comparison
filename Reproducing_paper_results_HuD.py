import scipy.optimize as opt
import numpy as np
import math
import keras
from keras.models import load_model
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams

M = 25 #49#64
N = 64 #49
pi=np.pi

theta_= np.linspace(0,pi/2,90)
phi_= np.linspace(0,2*pi,360)
theta,phi = np.meshgrid(theta_,phi_) # Makind 2D grid

#df_element0 = pd.read_excel('./Coding_Metamaterials/Reflection_phase_of_elements_with_more_frequency_points.xlsx', sheet_name = 'Element0') 
#df_element1 = pd.read_excel('./Coding_Metamaterials/Reflection_phase_of_elements_with_more_frequency_points.xlsx', sheet_name = 'Element1') 

#df_element0["reflectionphase_unwrapped"] = np.unwrap(np.deg2rad(df_element0["reflectionphase"]))
#df_element1["reflectionphase_unwrapped"] = np.unwrap(np.deg2rad(df_element1["reflectionphase"]))

df_element0 = pd.read_excel('./wideband_coding_metasurface/Reflectionphase_of_element.xlsx', sheet_name = 'Unit0') 
df_element1 = pd.read_excel('./wideband_coding_metasurface/Reflectionphase_of_element.xlsx', sheet_name = 'Unit1') 

def fun(x,i):
    element = x
    phase_pred = []
    for j in element:
        if(j == 0):
            phase_pred.append(df_element0["reflectionphase_unwrapped"][i])
        if(j == 1):
            phase_pred.append(df_element1["reflectionphase_unwrapped"][i])   
    print(phase_pred)
        #omsriramajayam
    reflection_phase = np.reshape(phase_pred, (M,N))
    result = []
    S = 0
    for m in range(M):
       for n in range(N):    
            S =  S + np.exp(-1j * (reflection_phase[m,n] + k[i]*D[i]*np.sin(theta)*((m-1/2)*np.cos(phi)+((n-1/2)*np.sin(phi)))))
    S = S
    H = np.trapz(np.trapz(np.abs(S)**2*np.sin(theta),theta_),phi_) # integration using trapezoid function
    directivity = 4 * pi * np.abs(S)**2 / H
    rcs = (1/(4*pi*N**2)) * np.max(directivity)  
    result.append(10 * np.log10(rcs)) 
        #frequency.append(df["frequency"][i])
    return result    
	
rcs_over_frequency = {}#pd.DataFrame([])
list_of_rcs_over_frequency = []
list_for_many_combinations = pd.DataFrame([])
number_of_frequency_points = 397 #for wideband coding metasurface #685 for coding metamaterials

lambda0 = np.zeros(number_of_frequency_points)
k = np.zeros(number_of_frequency_points)
D = np.zeros(number_of_frequency_points)

dataframe = pd.read_excel('Coding_metamaterial_matrix_5by5supercell_Nequals8.xlsx', header = None)#pd.read_excel('HuD_10points_supercell_repeated_complete.xlsx', header = None)

x = np.zeros((1,1600))#((1,3136))
#len(df_frequency)
for times in range(dataframe.shape[0]):  
    for i in range(number_of_frequency_points):  
        lambda0[i] = (3.00*10**8)/(df_element0["frequency"][i])
        k[i] = 2*pi/lambda0[i]
        D[i] = lambda0[i]
        x[times] = dataframe.iloc[times].to_numpy()#np.array((t,l)).ravel() 
        #x[times][N**2:] = dataframe.iloc[times,N**2:].to_numpy()   
        rcs_over_frequency['Combination_number_%d' %times] = fun(x[times],i)
        list_of_rcs_over_frequency.extend(rcs_over_frequency['Combination_number_%d' %times])
    
for j in range(dataframe.shape[0]):
    list_for_many_combinations['Combination_number_%d' %j] = list_of_rcs_over_frequency[number_of_frequency_points*j:number_of_frequency_points*j+number_of_frequency_points]#[len(df_v1)*j:len(df_v1)*j+len(df_v1)]

#list_for_many_combinations.to_excel("RCS_over_all_frequencies_for_all_combinations.xlsx") 

for k in range(list_for_many_combinations.shape[1]):
    plt.figure()
    plt.xlabel("Frequency GHz")
    plt.ylabel("RCS in dB")
    plt.title("RCS for %d combination of 0 and 1 elements from 7GHz to 14GHz \n" %k, loc = 'right')
    plt.plot(df_element0["frequency"][0:number_of_frequency_points],list_for_many_combinations['Combination_number_%d' %k])
    plt.savefig("RCS over frequency for Combination of 5by5 supercell and N = 8_%d_%%d.png" %k %number_of_frequency_points)
plt.show()
plt.ion() # helps to come to next line in command window without closing figures

for k in range(list_for_many_combinations.shape[1]):
    plt.figure()
    plt.xlabel("Frequency GHz")
    plt.ylabel("RCS reduction in dB")
    plt.ylim(-30, 0)
    plt.yticks(np.arange(-30,10, 10.0))
    plt.xlim(7.5,13)
    plt.xticks(np.arange(7.5,13.5, 0.5))
    
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams["font.weight"] = "bold"
    
    plt.title("RCS for %d combination of 0 and 1 elements from 7GHz to 13GHz \n" %k, loc = 'center', fontweight="bold")
    plt.plot(df_element0["frequency"][0:number_of_frequency_points],list_for_many_combinations['Combination_number_%d' %k], color = 'r', label = 'RCS reduction value over frequency')
    plt.legend(['RCS reduction value over frequency'])
    plt.grid(True)
    
    plt.savefig("Well Formatted RCS over frequency for Combination of 5by5 supercell and N = 8_%d_%%d.png" %k %number_of_frequency_points, bbox_inches='tight')
    plt.show()
	
