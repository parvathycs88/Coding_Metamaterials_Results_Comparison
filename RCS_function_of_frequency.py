import scipy.optimize as opt
import numpy as np
import math
import keras
from keras.models import load_model
import pandas as pd
import time
import matplotlib.pyplot as plt

N = 8
pi=np.pi

theta_= np.linspace(0,pi/2,90)
phi_= np.linspace(0,2*pi,360)
theta,phi = np.meshgrid(theta_,phi_) # Makind 2D grid

df_frequency = pd.read_excel('Frequency_range.xlsx')
#df_v1 = pd.read_excel('ReflectionPhase_1_openingangle_10_length_6.xlsx') # dimensions of '0'th element V1
#df_v2 = pd.read_excel('ReflectionPhase_1_openingangle_85_length_10.xlsx') # dimensions of '0'th element V2
lambda0 = pd.DataFrame([])
lambda0 = 1/(df_frequency["frequency"].div(3*10^8))

k = 2*pi/lambda0
D = lambda0


#df_v1["reflectionphase_unwrapped"] = np.unwrap((np.deg2rad(df_v1["reflectionphase"])) % 2*np.pi) # modulo 2*pi helps to change range from -pi to +pi to 0 to 2*pi
#df_v2["reflectionphase_unwrapped"] = np.unwrap((np.deg2rad(df_v2["reflectionphase"])) % 2*np.pi)

#omsriramajayam

def fun(x,i):
    #L_v = x[:N**2]
    #theta_v = x[N**2:]
    element = x[:N**2]
    phase_pred = []
    for j in element:
        if(j == 0):
            phase_pred.append(0)
        if(j == 1):
            phase_pred.append(np.pi)   
    print(phase_pred)
        #omsriramajayam
    reflection_phase = np.reshape(phase_pred, (N,N))
    #phase_pred = np.repeat(df["reflectionphase_in_radians"][i], N**2)
    #reflection_phase = np.reshape(phase_pred, (N,N))
    result = []
    S = 0
    for m in range(N):
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

dataframe = pd.read_excel('Coding_metamaterials_matrix.xlsx', header = None)
#x = np.zeros((99,200))  #Initialise numpy array x

    #lambda0[i] = 3*10^8/(df["frequency"][i])
    #k[i] = 2*pi/lambda0[i]
    #D[i] = lambda0[i]#0.03#d=0.015 used for simulation by haoyang #lambda0/2
x = np.zeros((1,100))  #Initialise numpy array x
number_of_frequency_points = 100#len(df_frequency)
for times in range(dataframe.shape[0]):  
    for i in range(number_of_frequency_points):  
        x[times][:N**2] = dataframe.iloc[times,:N**2].to_numpy()#np.array((t,l)).ravel() 
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
    plt.title("RCS for %d combination of 0 and 1 elements from 6GHz to 14GHz \n" %k, loc = 'right')
    plt.plot(df_frequency["frequency"][0:number_of_frequency_points],list_for_many_combinations['Combination_number_%d' %k])
    plt.savefig("RCS over frequency for Combination_number_%d_%%d.png" %k %number_of_frequency_points)
plt.show()
plt.ion() # helps to come to next line in command window without cosing figures


   
