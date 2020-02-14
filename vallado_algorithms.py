# -*- coding: utf-8 -*-
'''
Created on Mon Dec 16 21:33:31 2019
@author: sam
'''

import numpy as np
import sympy as sp


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
def KepEqtnP(mu, t, p):
    '''
    Solves Kepler's Equation for the Parabolic Anomaly
    Args:
        mu - gravitational parameter of primary body
        t - change in time in seconds
        p - semiparameter in kilometers
    Returns:
        B - parabolic anomaly in radians
    '''
    
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


# Algorithm 7
def KeplerCOE(mu, state, t):
    '''
    Determines position and velocity after a change in time
    Args:
        mu - gravitational parameter of primary body
        state - initial state vector
        t - change in time
    Returns
        staten - state vector following change in time
    '''
    
    r = state[:3]
    
    h, E, n, e, p, a, i, OM, om, v = RV2COE(state)
    
    if e != 0:
        An = vtoAnomaly(e, v)
    else:
        An = v
        
    # elliptic case
    if e < 1.0:
        M = An - e*np.sin(An)
        Mn =  M + n*t
        E = KepEqtnE(Mn, e)
        An_n = E
    
    # parabolic case
    elif e == 1.0:
        hvec = np.linalg.cross(r, v)
        h = np.linalg.norm(hvec)
        p = h**2/mu # why do we do this if we already used RV2COE above?
        M = An + (An**3/3)
        E = KepEqtnP(mu, t, p)
    
    # hyperbolic case
    else:
        M = e*np.sinh(An) - An
        Mn = M + n*t
        E = KepEqtnH(Mn, e)
        
    if e != 0:
        v = Anomalytov(e, An_n)
    else:
        v = E
        
    staten = COE2RV(mu, p, e, i, OM, om, v)
    
    return staten


# Algorithm 8
def Kepler(mu, initial_state, delta_t):
    '''
    Determines state of satellite after a change in time
    Args:
        mu - gravitational parameter of the primary body
        initial_state - initial state vector
        delta_t - time of flight in seconds
    Returns:
        state - subsequent state vector
    '''
    
    r0 = initial_state[:3]
    v0 = initial_state[3:]
    alpha = -(np.linalg.norm(v0)**2)/mu + 2/np.linalg.norm(r0)
    tol = 0.000001
    
    # circle or ellipse
    if alpha > tol:
        x0 = np.sqrt(mu)*delta_t*alpha
     
    # parabola
    elif abs(alpha) < tol:
        h_vec = np.cross(r0, v0)
        p = np.linalg.norm(h_vec)**2/mu
        s = 2*sp.acot(3*np.sqrt(mu/p**3)*(delta_t))
        w = sp.atan(np.tan(s))**(1/3)
        x0 = np.sqrt(p)*2*sp.cot(2*w)
        
    # hyperbola
    elif alpha < -tol:
        a = 1/alpha
        x0 = np.sign(delta_t)*np.sqrt(-a)*np.log((-2*mu*alpha*delta_t)/
                     (np.dot(r0, v0) + np.sign(delta_t)*np.sqrt(-mu*a)*
                     (1 - np.linalg.norm(r0)*alpha)))
    
    xn = x0 + 1
    while abs(xn - x0) > 10**-6:
        psi = xn**2*alpha
        c2, c3 = find_c2c3(psi)
        r = (xn**2*c2 + (np.dot(r0, v0)/np.sqrt(mu))*xn*(1 - psi*c3) 
                    + np.linalg.norm(r0)*(1 - psi*c2))
        xn1 = xn + ((np.sqrt(mu)*delta_t - xn**3*c3 - 
                     (np.dot(r0, v0)/np.sqrt(mu))*xn**2*c2 - 
                     np.linalg.norm(r0)*xn*(1 - psi*c3))/r)
        x0 = xn
        xn = xn1
    
    f = 1 - (xn**2/np.linalg.norm(r0))*c2
    g = delta_t - (xn**3/np.sqrt(mu))*c3
    gdot = 1 - (xn**2/r)*c2
    fdot = (np.sqrt(mu)/(r*np.linalg.norm(r0)))*xn*(psi*c3 - 1)
    r_vec = f*r0 + g*v0
    v_vec = fdot*r0 + gdot*v0
    state = np.append(r_vec, v_vec)
    
    return state  


# Algorithm 9
def RV2COE(mu, state):
    '''
    Converts a state vector to the classical orbital elements
    * does not include special cases *
    Args:
        mu - gravitational parameter of primary body
        state - position and velocity as a 6 element array
    Returns:
        h_mag - specific angular momentum
        E - specific mechanical energy 
        n_mag - magnitude of the node vector 
        e_mag - ecentricity 
        p - semiparameter 
        a - semimajor axis 
        i_deg - inclination in radians 
        Omega_deg - longitude of the ascending node in radians 
        omega_deg - argument of perigee in radians 
        true_deg - true anomally in radians
    '''
    
    tol = 1*10**-6
    K = [0, 0, 1]
    
    # position vector
    r = state[:3]
    r_mag = np.linalg.norm(r)
    
    # velocity vector
    v = state[3:]
    v_mag = np.linalg.norm(v)
    
    # specific angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    
    # node vector
    n = np.cross(K, h)
    n_mag = np.linalg.norm(n)
    
    # eccentricity
    e = ((v_mag**2 - (mu/r_mag))*r - np.dot(r, v)*v)/mu
    e_mag = np.linalg.norm(e)
    
    # specific energy
    E = (v_mag**2/2) - (mu/r_mag)
    
    # semiparameter and semimajor axis depending on orbit type
    if 1 - tol < e_mag < 1 + tol:
        p = (h_mag**2/mu)
        a = np.inf
    else:
        a = -mu/(2*E)
        p = a*(1 - e_mag**2)
        
    # inclination
    i = np.arccos(h[2]/h_mag)
    
    # longitude of the ascending node
    Omega = np.arccos(n[0]/n_mag)
    if n[1] < tol:
        Omega = 2*np.pi - Omega
    
    # argument of perigee
    omega = np.arccos((np.dot(n, e))/(n_mag*e_mag))
    if e[2] < 0:
        omega = 2*np.pi - omega
       
    # true anomaly
    true = np.arccos(np.dot(e, r)/(e_mag*r_mag))
    if np.dot(r, v) < 0:
        true = 2*np.pi - true
    
    return h_mag, E, n_mag, e_mag, p, a, i, Omega, omega, true
    

# Algorithm 10  
def COE2RV(mu, p, e, i, Omega, omega, true):
    '''
    Converts the classical orbital elements to a state vector
    * does not include special cases *
    Args:
        mu - gravitational parameter of primary body
        p - semiparameter 
        e - ecentricity  
        i - inclination in radians 
        Omega - longitude of the ascending node in radians 
        omega - argument of perigee in radians 
        true - true anomally in radians
    Returns:
        state - state vector
    '''
    
    # position and velocity PQW vectors
    r_pqw = np.matrix([[(p*np.cos(true))/(1 + e*np.cos(true))],
                       [(p*np.sin(true))/(1 + e*np.cos(true))], 
                       [0]])
    v_pqw = np.matrix([[-np.sqrt(mu/p)*np.sin(true)],
                       [np.sqrt(mu/p)*(e + np.cos(true))], 
                       [0]])
    
    # PQW to IJK transformation matrix
    pqw2ijk = np.matrix([[ np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.sin(omega)*np.cos(i),
                          -np.cos(Omega)*np.sin(omega) - np.sin(Omega)*np.cos(omega)*np.cos(i),
                           np.sin(Omega)*np.sin(i)],
                         [ np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.sin(omega)*np.cos(i),
                          -np.sin(Omega)*np.sin(omega) + np.cos(Omega)*np.cos(omega)*np.cos(i),
                          -np.cos(Omega)*np.sin(i)],
                         [ np.sin(omega)*np.sin(i),
                           np.cos(omega)*np.sin(i),
                           np.cos(i)]])
    
    # position and velocity IJK vectors
    r_ijk = pqw2ijk*r_pqw
    v_ijk = pqw2ijk*v_pqw
    
    # convert to a single state vector
    r = r_ijk.getA1()          
    v = v_ijk.getA1()
    state = np.concatenate((r, v))
    
    return state


# Algorithm 11
def findTOF(mu, r0, r, p):
    '''
    Determines a satellite's time of flight   
    Args:
        mu - gravitational parameter of primary body
        r0 - initial position
        r - current position
        p - semiparameter
    Returns:
        tof - time of flight
    '''
    
    delta_v = np.arccos((np.dot(r0, r)/(np.linalg.norm(r0)*np.linalg.norm(r))))
    k = np.linalg.norm(r0)*np.linalg.norm(r)*(1 - np.cos(delta_v))
    l = np.linalg.norm(r0) + np.linalg.norm(r)
    m = np.linalg.norm(r0)*np.linalg.norm(r)*(1 + np.cos(delta_v))
    a = (m*k*p)/((2*m - l**2)*p**2 + 2*k*l*p - k**2)
    f = 1 - (r/p)*(1 - np.cos(delta_v))
    g = (np.linalg.norm(r0)*np.linalg.norm(r)*np.sin(delta_v))/np.sqrt(mu*p)
    
    # elliptical
    if a > 0.0000001:
        fdot = np.sqrt(mu/p)*np.tan(delta_v/2)*(((1 - np.cos(delta_v))/p) - 
                       (1/np.linalg.norm(r0)) - (1/np.linalg.norm(r)))
        E = np.arcsin((-np.linalg.norm(r0)*np.linalg.norm(r)*fdot)/np.sqrt(mu*a))
        tof = g + np.sqrt(a**3/mu)*(E - np.fin(E))
    
    # parabolic
    elif a > 999999999999:
        c = np.sqrt(np.linalg.norm(r0)**2 + np.linalg.norm(r)**2 - 
                    2*np.linalg.norm(r0)*np.linalg.norm(r)*np.cos(delta_v))
        s = (np.linalg.norm(r0) +np.linalg.norm(r) + c)/2
        tof = (2/3)*np.sqrt(s**3/(2*mu))*(1 - ((s - c)/s)**(3/2))
    
    #hyperbolic
    elif a < 0.0:
        H = np.arccosh(1 + (f - 1)*(np.linalg.norm(r0)/a))
        tof = g + np.sqrt((-a)**3/mu)*(np.sinh(H) - H)
        
    return tof


# Algorithm 12
def ECEFtoLatLon(mu, r, r_ECEF):
    '''
    Converts from ECEF coordinates to latitude and longitude
    Args:
        mu - gravitational parameter fo the primary body
        r - radius of the primary body?
        r_ECEF - position in ECEF coordinates
    Returns:
        phi - geodetic latitude
        long - longitude
        h - height
    '''
    
    r = np.sqrt(r_ECEF[0]**2 + r_ECEF[1]**2)
    alpha = np.arcsin(r_ECEF[1]/r)
    long = alpha
    delta = np.arcsin(r_ECEF[2]/np.linalg.norm(r_ECEF))
    phi_gd = delta
    r_delta = r
    rk = r_ECEF[2]
    phi_gdold = phi_gd -1
    
    while phi_gd - phi_gdold < 10**-8:
        C = (np.tan(phi_gd)*r_delta - rk)/()
    
    
# Algorithm 36
def HohmannTransfer(mu, r, r0, rf):
    '''
    Calculates the necessary parameters to perform a Hohmann Transfer
    Args:
        mu - gravitational parameter of the primary body
        r - radius of primary body in km
        r0 - initial radius in km
        rf - final radius in km
    Returns:
        a_trans - transfer orbit semimajor axis in km
        t_trans - 
        delta_va - 
        delta_vb - 
    '''
    
    r1 = r0/r
    r2 = rf/r
    a_transfer = (r0 + rf)/2
    a_trans = (r1 + r2)/2
    v0 = np.sqrt(mu/r0)
    vf = np.sqrt(mu/rf)
    v_transa = np.sqrt((2*mu/r0) - (mu/a_trans))
    v_transb = np.sqrt((2*mu/rf) - (mu/a_trans))
    delta_va = v_transa - v0
    delta_vb = vf - v_transb
    t_trans = np.pi*np.sqrt(a_trans**3/mu)
    
    return a_transfer, t_trans, delta_va, delta_vb


# Algorithm 37
def BiEllipticTransfer(r0, rb, rf):
    '''
    Calculates the necessary parameters to perform a Bi-Elliptic Transfer
    Args:
        mu - gravitational parameter of the primary body
        r0 - initial radius
        rb - intermediate radius
        rf - final radius
    Returns:
        a_trans1 - first transfer orbit semimajor axis
        a_trans2 - second transfer orbit semimajor axis
        t_trans -  
        delta_va - 
        delta_vb - 
        delta_vc - 
    '''
    
    a_trans1 = (r0 + rb)/2
    a_trans2 = (rb + rf)/2
    v0 = np.sqrt(mu/r0)
    v_trans1a = np.sqrt(2*mu/r0 - mu/a_trans1)
    v_trans1b = np.sqrt(2*mu/rb - mu/a_trans1)
    v_trans2b = np.sqrt(2*mu/rb - mu/a_trans2)
    v_trans2c = np.sqrt(2*mu/rf - mu/a_trans2)
    vf = np.sqrt(mu/rf)
    
    delta_va = v_trans1a - v0
    delta_vb = v_trans2b - v_trans1b
    delta_vc = vf - v_trans2c
    t_trans = np.pi*np.sqrt(a_trans1**3/mu) + np.pi*np.sqrt(a_trans2**3/mu)
    
    return a_trans1, a_trans2, t_trans, delta_va, delta_vb, delta_vc


# Algorithm 38
def OneTangentBurn_Perigee(mu, r0, rf, v_transb):
    '''
    Computes paramenters for the one-tangent burn for elliptical orbits at perigee
    Args:
        mu - gravitational parameter of primary body
        r0 - initial radius
        rf - final radius
        v_transb - transfer point
    Returns:
        e_trans
        a_trans
        t_trans
        delta_va
        delta_vb
    '''
    R = r0/rf
    e_trans = (R - 1)/(np.cos(v_transb) - R)
    a_trans = r0/(1 - e_trans)
    v0 = np.sqrt(mu/r0)
    v_transa = np.sqrt((2*mu/r0) - (mu/a_trans))
    vf = np.sqrt(mu/rf)
    v_transb = np.sqrt((2*mu/rf) - (mu/a_trans))
    delta_va = v_transa - v0
    
    phi_transb = np.arctan((e_trans*np.sin(v_transb))/
                           (1 + e_trans*np.cos(v_transb)))
    delta_vb = np.sqrt(v_transb**2 + vf**2 - 2*v_transb*vf*np.cos(phi_transb))
    E = np.arccos((e_trans + np.cos(v_transb))/(1 + e_trans*np.cos(v_transb)))
    t_trans = np.sqrt(a_trans**3/mu)*((E - e_trans*np.sin(E)))
    
    return e_trans, a_trans, t_trans, delta_va, delta_vb


# Algorithm 39
def InclinationOnly(delta_i, fpa, v0):
    '''
    Simple inclination change
    Args:
        delta_i - change in inclination in radians
        fpa - flight path angle in radians
        v0 - initial velocity
    Returns:
        delta_vi - change in velocity
    '''
    
    delta_vi = 2*v0*np.cos(fpa)*np.sin(delta_i/2)
    
    return delta_vi


# Algorithm 40
def ChangeinAscendingNode(delta_Omega, i0, v0):
    '''
    Change in Ascending Node for circular orbits
    Args:
        delta_Omega - change in ascending node
        i0 - initial inclination
        v0 - initial velocity
    Returns:
        delta_v - change in velocity
    '''
    
    u = np.arccos(np.cos(i0)**2 + np.sin(i0)**2*np.cos(delta_Omega))
    delta_v = 2*v0*np.sin(u/2)
    
    return delta_v


# Algorithm 41
def CombinedChangestoiandOmega(i0, i_f, delta_Omega, v0):
    '''
    Combined changes to inclination and Ascending Node for circular orbits
    Args:
        i0 - initial inclination in radians
        i_f - final inclination in radians
        delta_Omega - change in ascending node in radians
        v0 - initial velocity
    Returns:
        delta_v - change in velocity
    '''
    
    u = np.arccos(np.cos(i0)*np.cos(i_f) + 
                  np.sin(i0)*np.sin(i_f)*np.cos(delta_Omega))
    delta_v = 2*v0*np.sin(u/2)
    
    return delta_v

# Algorithm 42
def MinCombinedPlaneChange(r0, rf, delta_i):
    '''
    Computes the change in inclination for a combined plane change
    Args:
        r0 - initial radius
        rf - final radius
        delta_i = change in velocity
    Returns:
        delta_i0 - change in initial inclination
        delta_if - change in final inclination
    '''
    
    R = rf/r0
    s = (1/delta_i)*np.arctan(np.sin(delta_i)/(R**(3/2) + np.arccos(delta_i)))
    delta_i0 = s*delta_i
    delta_if = (1 - s)*delta_i
    
    return delta_i0, delta_if

# Algorithm 43
def FixedDeltavManeuver(v0, vf, v_transa):
    '''
    Computes the payload angle and inclination change for a fixed 
    delta v maneuver
    Args:
        v0 - initial velocity
        vf - final velocity
        v_transa - transfer velocity
        delta_v - change in velocity
    Returns:
        gamma - payload angle
        delta_i - change in inclination
    '''
    
    delta_v = vf - v0
    gamma = np.arccos(-(v0**2 + delta_v**2 - vf**2)/(2*v0*delta_v))
    
    # determine if the inclination is increasing or decreasing
    if -np.pi/2 <= gamma <= 0:
        i = -1
    elif 0< gamma < np.pi/2:
        i = 1
    
    delta_i = i*np.arccos((v0**2 + v_trans**2 - delta_v**2)/(2*v0*v_transa))
    
    return delta_i

# Algorithm 44
def CircularCoplanarPhasing(a_tgt, nu, k):
    '''
    Circular coplanar phasing for same orbits
    Args:
        a_tgt - target semimajor axis ratio
        nu - phase angle measured from the target to the interceptor
        k - target revolutions
    Returns:
        tau - period
        delta_v - change in velocity
        a_phase - semimajor axis
    '''
    
    mu = 1
    omega = np.sqrt(mu/(a_tgt**3))
    tau = (2*np.pi*k + nu)/omega
    a_phase = (mu*((tau/(2*np.pi*k))**2))**(1/3)
    delta_v = 2*abs(np.sqrt((2*mu/a_tgt) - (mu/a_phase)) - np.sqrt(mu/a_tgt))
    
    return tau, delta_v, a_phase

# Algorithm 45
def CircularCoplanarPhasing2(nu0, a_int, a_tgt, k):
    '''
    Circular coplanar phasing for different orbits
    Args:
        nu0 - initial phase angle measured from the target to the interceptor
        a_int - initial ratio
        a_tgt - target ratio
        k
    Returns:
        tau_trans - period
        delta_v - change in velocity
        a_trans - semimajor axis
    '''
    
    mu = 1
    omega_tgt = np.sqrt(mu/a_tgt**3)
    omega_int = np.sqrt(mu/a_int**3)
    a_trans = (a_int + a_tgt)/2
    tau_trans = np.pi*np.sqrt(a_trans**3/mu)
    alpha = omega_tgt*tau_trans
    nu = alpha - np.pi
    tau_wait = (nu - nu0 + 2*np.pi*k)/(omega_int - omega_tgt)
    delta_v = (abs(np.sqrt((2*mu/a_int) - (mu/a_trans)) - np.sqrt(mu/a_int)) +
               abs(np.sqrt((2*mu/a_tgt) - (mu/a_trans)) - np.sqrt(mu/a_tgt)))
    
    return tau_trans, delta_v, a_trans

# Algorithm 46
def NoncoplanarPhasing(nu0, a_int, a_tgt, k_tgt, u_int, Omega, true0, delta_i):
    ''' 
    Noncoplanar phasing
    Returns:
        tau_trans
        tau_total
        delta_vphase
        delta_vtrans1
        delta_vtrans2
        a_phase
    '''
    
    omega_tgt = np.sqrt(mu/a_tgt**3)
    omega_int = np.sqrt(mu/a_int**3)
    a_trans = (a_int + a_tgt)/2
    tau_trans = np.pi*np.sqrt(a_trans**3/mu)
    alpha = omega_tgt*tau_trans
    delta_nuint = np.pi*2 - u_int
    delta_tnode = delta_nuint/omega_int
    true_tgt1 = true0 + omega_tgt*delta_tnode
    true_int1 = Omega + np.pi
    nu_new = true_int1 - true_tgt1
    alpha_new = np.pi + nu_new
    P_phase = (alpha_new - alpha + 2*np.pi*k_tgt)/omega_tgt
    a_phase = (mu*(P_phase/(k_tgt*2*np.pi))**2)**(1/3)
    delta_vphase = abs(np.sqrt((2*mu/a_int) - (mu/a_phase)) - np.sqrt(mu/a_int))
    delta_vtrans1 = abs(np.sqrt((2*mu/a_int) - (mu/a_trans)) - 
                        np.sqrt((2*mu/a_int) - (mu/a_phase)))
    delta_vtrans2 = abs(np.sqrt(((2*mu/a_tgt) - (mu/a_trans)) + (mu/a_tgt) - 
                                a*np.sqrt((2*mu/a_tgt) - 
                                (mu/a_trans))*np.sqrt(mu/a_tgt)*np.cos(delta_i)))
    tau_total = 2*np.pi*np.sqrt(a_phase**3/mu) + tau_trans + delta_tnode
    
    return tau_trans, tau_total, delta_vphase, delta_vtrans1, delta_vtrans2, a_phase


# Algorithm 59
def Lambert_BattinMethod(mu, r0, r, dt, tm = 1, orbit_type = 1):
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
    m = (mu*dt**2)/(8*r_op**3)
    
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

    
def Algorithm_BPlane(state, v_inf):
    '''
    
    '''
#  cal state over a period
#    convert states to long, lat, alt
#    plot      
#    regex replace strings in deck
#    scipy optimize           

        
        
if __name__ == '__main__':

    mu = 398600.4415
##    r_earth = 6378.137    
##    # Algorithms 9 and 10
#    statevec = np.array([6524.834, 6862.875, 6448.296,
#                         4.901327, 5.533756,-1.976341])
#    h, E, n, e, p, a, i, Omega, omega, true = RV2COE(mu, statevec)
#    print(' ')
#    print('Position: {} km \
#          \nVelocity: {} km/s \
#          '.format(statevec[:3], statevec[3:]))
#    print(' ')
#    print('Specific angular momentum: {} km^2/s \
#          \nSpecific mechanical energy: {} J \
#          \nNode magnitude: {} km^2/s \
#          \nEccentricity magnitude: {} \
#          \nSemiparameter: {} km \
#          \nSemimajor axis: {} km \
#          \nInclination: {} deg \
#          \nLongitude of the ascending node: {} deg \
#          \nArgument of perigee: {} deg \
#          \nTrue anomally: {} deg \
#          '.format(h, E, n, e, p, a, i*(180/np.pi), 
#                   Omega*(180/np.pi), omega*(180/np.pi), true*(180/np.pi)))
#    print(' ')
    r0 = np.array([15945.34, 0, 0])
    r = np.array([12214.83899, 10249.46731, 0])
    delta_t = 76*60
    testttt = Lambert_BattinMethod(mu, r0, r, delta_t)
    print(testttt)
#    test999 = lambert_battin(mu, r0, r, delta_t)
#    print(test999)
#    state = COE2RV(mu, p, e, i, Omega, omega, true)
#    
#    # Algorithm 8
#    state2 = np.array([1131.340, -2282.343, 6672.423,
#                       -5.64305,   4.30333, 2.42879])
#    tof = 40*60
#    result = Kepler(mu, state2, tof)
#    print(result)
#    
#    # Algorithm 37
#    r0 = 6569.4781
#    rf = 42159.4856
#    res = HohmannTransfer(mu, r_earth, r0, rf)
#    print(res)
#    
#    # Example 2-4
#    state = np.array([1131.340, -2282.343, 6672.42,
#                      -5.64305,   4.30333, 2.42879])
#    delta_t = 40*60
#    # FIND: position and velocity at future time
#    new_state = Kepler(mu, state, delta_t)
#    print('Example 2-4:')
#    print('Position: {} km'.format(new_state[:3]))
#    print('Velocity: {} km/s'.format(new_state[3:]))