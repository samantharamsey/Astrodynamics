# -*- coding: utf-8 -*-
'''
Created on Mon Oct 28 23:05:35 2019
@author: sam
'''

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math


# Algorithm 1
def find_c2c3(psi):
    '''
    Determines c2 and c3 values to solve Keplers Equation
    Args:
        psi - in radians
    Returns:
        c2
        c3
    '''
    
    if psi > 1*10**-6:
        c2 = (1 - np.cos(np.sqrt(psi)))/psi
        c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi)))/np.sqrt(psi**3) 
        
    else:
        if psi < 1*10**-6:
            c2 = (1 - np.cosh(np.sqrt(-psi)))/psi
            c3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi))/np.sqrt((-psi)**3)
        else:
            c2 = 1/2
            c3 = 1/6
            
    return c2, c3


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


# Algorithm 3
def KepEqtnP(t, p):
    '''
    Solves Kepler's Equation for the Parabolic Anomaly
    Args:
        t - change in time in seconds
        p - semiparameter in kilometers
    Returns:
        B - parabolic anomaly in radians
    '''
    
    mu = 3.98574405096*10**5
    n = 2*np.sqrt(mu/(p**3))
    s = (1/2)*(sp.acot(3/2)*n*t)
    w = sp.atan((sp.tan(s))**(1/3))
    B = 2*sp.cot(2*w)
    
    return B
    

# Algorithm 4
def KepEqtnH(M, e, tolerance):
    '''
    Solves Kepler's Equation for Hyperbolas
    Args:
        M - Mean Anomaly
        e - eccentricity
        tolerance - tolerance for convergence
    Returns:
        H - Hyperbolic Anomaly
    '''
    if e < 1.6:
        if -np.pi < M < 0 or M > np.pi:
            H = M - e
        else:
            H = M + e
    else:
        if e < 3.6 and np.abs(M) > np.pi:
            H = M - M*e
        else:
            H = M/(e -1)
    
    residual = 1
    while residual > tolerance:
        Hn = H + ((M - e*np.sinh(H) + H)/(e*np.cosh(H) - 1))
        residual = np.abs(Hn - H)
        H = Hn
        
    return H


# Algorithm 5
def vtoAnomaly(e, v):
    '''
    Converts eccentricity and true anomaly to E, B, or H
    Args:
        e - eccentricity
        v - true anomaly
    Returns:
        An - Eccentric, Parabolic, or Hyperbolic Anomaly
    '''
    
    if e < 1.0:
        An = np.arcsin((np.sin(v)*np.sqrt(1 - e**2))/(1 + e*np.cos(v)))
    elif e == 1.0:
        An = np.tan(v/2)
    else:
        An = np.arccosh((e + np.cos(v))/(1 + e*np.cos(v)))
        
    return An


# Algorithm 6
def Anomalytov(e, An, p = 1, r = 1):
    '''
    Converts Eccentric, Hyperbolic, or Parabolic Anomaly to True Anomaly
    Args:
        e - eccentricity
        An - Eccentric, Hyperbolic, or Parabolic Anomaly
        if Parabolic Anomaly:
            p - semiparameter
            r - radius
    Returns:
        v - True Anomaly
    '''
    
    if e < 1.0:
        v = np.arccos((np.cos(An) - e)/(1 - e*np.cos(An)))
    elif e == 1.0:
        v = np.arcsin((p*An)/r)
    else:
        v = np.arccos((np.cosh(An) - e)/(1 - e*np.cosh(An)))
        
    return v






def RV2COE(mu, state):
    ''' Converts from an initial state of position and velocity vectors to
        the classical orbital elements '''
    # tolerances
    zero = 0.00000001
    tol = 0.0000001
    # classical orbital elements
    rvec = state[:3] # r vector
    rmag = np.linalg.norm(rvec) # r magnitude
    vvec = state[3:] # velocity vector
    vmag = np.linalg.norm(vvec) # velocity magnitude
    hvec = np.cross(rvec, vvec) # angular momentum vector
    hmag = np.linalg.norm(hvec) # angular momentum magnitude
    k = np.array([0, 0, 1]) # unit vector in k direction
    nvec = np.cross(k, hvec) # node vector
    nmag = np.linalg.norm(nvec) # node magnitude
    evec = ((vmag**2 - (mu/rmag))*rvec - np.dot(rvec, vvec)*vvec) / mu # eccentricity vector
    emag = np.linalg.norm(evec) # eccentricity magnitude
    if emag < zero:
        emag = 0
    sme = vmag**2/2 - mu/rmag # specific mechanical energy
    if emag != 1:
        a = -(mu/(2*sme)) # semi major axis
        p = a*(1 - emag**2) # semiparameter: semi-latus rectum or ellipse
    else:
        a = math.inf # a = infinity
        p = hmag**2/mu
    i = np.arccos(hvec[2]/hmag) # inclination
    omega = np.arccos(nvec[0]/nmag)
    if nvec[1] < 0:
        omega = 2*np.pi - omega
    om = np.arccos((np.dot(nvec, evec))/(nmag*emag))
    if evec[2] < 0:
        om = 2*np.pi - om
    an = np.arccos((np.dot(evec, rvec))/(emag*rmag)) # true anomaly
    if np.dot(rvec, vvec) < 0:
        an = 2*np.pi - an
    # convert to degrees
    i = i*(180/np.pi)
    omega = omega*(180/np.pi)
    om = om*(180/np.pi)
    an = an*(180/np.pi)     
    # Special Cases  
    # Elliptical Equatorial
    if (zero < emag < 1) & (i == 0 or i == 180): # tolerance based
        omtrue = np.arccos(evec[0]/emag)
        if evec[1] < 0:
            omtrue = 2*np.pi - omtrue
        return {'angular momentum vector': hvec, 
                'angular momentum magnitude': hmag, 
                'node vector': nvec, 'node magnitude': nmag, 
                'eccentricity vector': evec, 'eccentricity magnitude': emag, 
                'specific mechanical energy': sme, 'semi-major axis': a, 
                'semi-latus rectum': p, 'inclination': i, 
                'right ascension of the ascending node': omega, 
                'argument of perigee': omtrue, 'true anomaly': an}
    # Circular Inclined
    elif (emag < zero) & (0-tol > i > tol):
        umag = np.arccos(np.dot(nvec, rvec)/(nmag*rmag))
        if rvec[2] < zero:
            umag = 2*np.pi - umag
        return {'angular momentum vector': hvec, 
                'angular momentum magnitude': hmag, 'node vector': nvec, 
                'node magnitude': nmag, 'eccentricity vector': evec, 
                'eccentricity magnitude': emag, 'specific mechanical energy': sme, 
                'semi-major axis': a, 'semi-latus rectum': p, 'inclination': i, 
                'right ascension of the ascending node': omega, 
                'argument of perigee': om, 'true an': umag}
    # Circular Equatorial
    elif (emag == 0) & (0-tol > i < tol): # +/- tolerance
        lamtrue = np.arccos(rvec[0]/rmag)
        if rvec[1] < zero:
            lamtrue = 2*np.pi - lamtrue
        return {'angular momentum vector': hvec, 
                'angular momentum magnitude': hmag, 'node vector': nvec, 
                'node magnitude': nmag, 'eccentricity vector': evec, 
                'eccentricity magnitude': emag, 'specific mechanical energy': sme, 
                'semi-major axis': a, 'semi-latus rectum': p, 'inclination': i, 
                'right ascension of the ascending node': omega, 
                'argument of perigee': om, 'lamda true': lamtrue}
    # if not special case
    else:
        return {'angular momentum vector': hvec, 
                'angular momentum magnitude': hmag, 'node vector': nvec, 
                'node magnitude': nmag, 'eccentricity vector': evec, 
                'eccentricity magnitude': emag, 'specific mechanical energy': sme, 
                'semi-major axis': a, 'semi-latus rectum': p, 'inclination': i, 
                'right ascension of the ascending node': omega, 
                'argument of perigee': om, 'true anomaly': an}
        
        
        
        