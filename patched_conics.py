# -*- coding: utf-8 -*-
'''
Created on Tue Feb 11 22:36:31 2020
@author: sam

Attempting a patched conic trajectory to Saturn
'''

import pandas as pd
import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from matplotlib import animation


############################# FURNSH THE KERNELS ##############################

def furnsh_kernels():
    spice.tkvrsn('TOOLKIT')
    path = r'C:/Spice_Kernels/'
    spice.furnsh(path + r'de430.bsp')
    spice.furnsh(path + r'naif0009.tls')
    spice.furnsh(path + r'sat425.bsp')
    spice.furnsh(path + r'cpck05Mar2004.tpc')
    spice.furnsh(path + r'020514_SE_SAT105.bsp')
    spice.furnsh(path + r'981005_PLTEPH-DE405S.bsp')
    spice.furnsh(path + r'030201AP_SK_SM546_T45.bsp')
    spice.furnsh(path + r'04135_04171pc_psiv2.bc')
    spice.furnsh(path + r'sat425.inp')
    spice.furnsh(path + r'sat425.cmt')

###############################################################################

def Lambert_BattinMethod(mu, r0, r, delta_t, tm = 1, orbit_type = 1):
    '''
    Lambert solver using Battin's method from fundamentals of astrodynamics 
    and applications - returns the two necessary delta v changes
    Args:
        mu - primary body gravitational parameter
        r0 - initial position
        r - final position
        delta_t - change in time
        tm - transfer method term
             +/- 1 depending on short/long term transfer
    Returns:
        v1 - delta velocity 1
        v2 - delta velocity 2
    '''
    
    r1 = np.linalg.norm(r0)
    r2 = np.linalg.norm(r)
    nu = np.arccos((np.dot(r0, r))/(r1*r2))
    if nu < 0:
        nu = nu + 2*np.pi
    c = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(nu))
    s = (r1 + r2 +c)/2
    epsilon = (r2 - r1)/r1
    w = (2*np.arctan((epsilon**2/4)/(np.sqrt(r2/r1) + 
                              r2/r1*(2 + np.sqrt(r2/r1)))))
    r_op = np.sqrt(r1*r2)*(np.cos(nu/4)**2 + np.tan(2*w)**2)
    m = (mu*delta_t**2)/(8*r_op**3)
    
    if 0 < nu < np.pi:
        l = ((np.sin(nu/4)**2 + np.tan(2*w)**2)/(np.sin(nu/4)**2
              + np.tan(2*w)**2 + np.cos(nu/2)))
    elif np.pi < nu < 2*np.pi:
        l = ((np.cos(nu/4)**2 + np.tan(2*w)**2 - np.cos(nu/2))/
             (np.cos(nu/4)**2 + np.tan(2*w)**2))
    else:
        print('error: invalid nu')
        breakpoint()
    
    if orbit_type == 1:
        x = l
    else:
        x = 0
        
    c1 = 8.0
    c2 = 1.0
    c3 = 9.0/7.0
    c4 = 16.0/63.0
    c5 = 25.0/99.0
    
    error = 10
    tolerance = 10**-6
    iter_max = 100
    iters = 0
    
    while error > tolerance:
        eta = x/(np.sqrt(1.0 + x) + 1.0)**2.0
        xi = c1*(np.sqrt(1 + x) + 1)/(3 + c2/(5 + eta 
                + c3*eta/(1 + c4*eta/(1 + c5*eta))))
        den = (1 + 2*x + l)*(4*x + xi*(3 + x))
        h1 = (l + x)**2*(1 + 3*x + xi)/den
        h2 = (m*(x - l + xi))/den
        B = 27*h2/(4*(1 + h1)**3)
        U = B/(2*(np.sqrt(1 + B) + 1))
        K = (1/3)/(1 + (4/27)*U/(1 + (8/27)*U/(1 + (700/2907)*U)))
        y = ((1 + h1)/3)*(2 + (np.sqrt(1 + B)/(1 + 2*U*K**2)))
        xn = np.sqrt(((1 - l)/2)**2 + (m/y**2)) - (1 + l)/2
        error = abs(x - xn)
        x = xn
        iters += 1
        if iters > iter_max:
            print('error: max iterations')
            breakpoint()
        
    a = (mu*delta_t**2)/(16*r_op**2*x*y**2)
    if a > 0.0:
        beta = 2*np.arcsin(np.sqrt((s - c)/(2*a)))
        if nu > np.pi:
            beta = beta*(-1)
        a_min = s/2
        t_min = np.sqrt(a_min**3/mu)*(np.pi - beta + np.sin(beta))
        alpha = 2*np.arcsin(np.sqrt(s/(2*a)))
        if delta_t > t_min:
            alpha = 2*np.pi - alpha
        delta_E = alpha - beta
        f = 1 - (a/r1)*(1 - np.cos(delta_E))
        g = delta_t - np.sqrt(a**3/mu)*(delta_E - np.sin(delta_E))
        g_dot = 1 - (a/r2)*(1 - np.cos(delta_t))
    else:
        alpha = 2*np.arcsinh(np.sqrt(s/(-2*a)))
        beta = 2*np.arcsinh(np.sqrt((s - c)/(-2*a)))
        delta_H = alpha - beta
        f = 1 - (a/r1)*(1 - np.cosh(delta_H))
        g = delta_t - np.sqrt(-a**3/mu)*(np.sinh(delta_H) - delta_H)
        g_dot = 1 - (a/r2)*(1 - np.cosh(delta_H))
    
    v1 = (r - f*r0)/g
    v2 = (g_dot*r - r0)/g
    
    return v1, v2

if __name__ == '__main__':
    
    # define some constants
    mu_sun = 132712440018 # km^3s^2
    mu_earth = 398600 # km^3s^2
    r_earth = 6378.137 # km
    
    # furnsh the SPICE kernels
    furnsh_kernels()
    
    # mission start and end dates
    utc = ['Jun 20, 2022', 'Dec 1, 2030']
    # convert to julian time
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])
    # difference in epoch
    delta_t = etTwo - etOne
    
    # find the positions
    earth_pos,  earth_lightTimes  = spice.spkpos('Earth', etOne, 'J2000', 
                                         'NONE', 'Solar System Barycenter')
    saturn_pos, saturn_lightTimes = spice.spkpos('Saturn', etTwo, 'J2000', 
                                         'NONE', 'Solar System Barycenter')
     
########################### INTERPLANETARY TRANSFER ###########################
    
    # lambert solver
    v1, v2 = Lambert_BattinMethod(mu_sun, earth_pos, saturn_pos, delta_t)
    
    # velocity of spacecraft relative to Earth at the SOI
    # need to update this to take quadrant into consideration
    # signs and trig functions are hard coded
    phi = np.arctan2(earth_pos[0], earth_pos[1])
    v_earth = 29.7859*np.array([-np.sin(phi), np.cos(phi), 0])
    # hyperbolic excess velocity
    v_inf_e = v1 - v_earth
    
    # velocity of spacecraft relative to Saturn at the SOI
    # need to update this to take quadrant into consideration
    # signs and trig functions are hard coded
    gamma = np.arctan2(saturn_pos[0], saturn_pos[1])
    v_saturn = 9.62724*np.array([np.cos(gamma), np.sin(gamma), 0])
    # hyperbolic excess velocity
    v_inf_s = v2 - v_saturn
    
############################ DEPARTURE FROM EARTH #############################

    # determine perigee altitude
    alt = 100 #km
    # energy equation for hyperbolic trajectory from earth
    epsilon = np.linalg.norm(v_inf_e)**2/2
    # required velocity at perigee
    v_p = np.sqrt(2*(mu_earth/(r_earth + alt) + epsilon))
    # circular parking orbit prior to interplanetary injection
    v_circular = np.sqrt(mu_earth/(r_earth + alt))
    # delta v to inject onto interplanetary orbit
    delta_v1 = v_p - v_circular
    
    
    
    
    