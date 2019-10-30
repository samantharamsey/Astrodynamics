# -*- coding: utf-8 -*-
'''
Created on Mon Oct 28 23:05:35 2019

@author: sam
'''

import numpy as np
import matplotlib.pyplot as plt

# Algorithm 2

def KepEqtnE(M, e, tolerance):
    '''
    Determines eccentric anomaly for Kepler's Equation
    Args:
        M - mean anomaly in radians
        e - eccentricity
        tolerance - tolerance for convergence
    Returns:
        En - eccentric anomaly
    '''

    if -np.pi < M < 0 or M > np.pi:
        E = M - e
    else:
        E = M + e
        
    residual = 1
    while residual > tolerance:
        En = E + ((M - E + e*np.sin(E))/(1 - e*np.cos(E)))
        residual = En - E
        E = En
        
    return En

# Recreate figure 2-6
tol = 10**-8
M = np.linspace(-np.pi, np.pi, 360)
e1, e2, e3 = 0.0, 0.5, 0.98

vec = np.vectorize(KepEqtnE)
E1 = vec([M], e1, tol)
E2 = vec([M], e2, tol)
E3 = vec([M], e3, tol)

plt.plot(E1)
plt.plot(E2)
plt.plot(E3)
plt.title('Figure 2-6')
plt.ylabel('Mean Anomaly')
plt.xlabel('Eccentric Anomaly')
plt.grid(True)
plt.legend(['e = 0.0', 'e = 0.5', 'e = 0.98'])
plt.show()

