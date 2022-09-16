"""Particle filter sensor and motion model implementations.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy.random import randn
from utils import gauss, wraptopi, angle_difference
import random

#Paricle filter motion model- Prediction of particle positions
def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt,gauss_std):

    M = particle_poses.shape[0]
    
    #defining motion model parameter variables
    phi1 = np.zeros(M)
    phi2 = np.zeros(M)
    d = np.zeros(M)

    #estimated local pose change equations 
    phi1Prime = np.arctan2((odom_pose[1]-odom_pose_prev[1]),(odom_pose[0]-odom_pose_prev[0])) - odom_pose_prev[2]
    dPrime = np.sqrt((odom_pose[1]-odom_pose_prev[1])**2 + (odom_pose[0]-odom_pose_prev[0])**2)
    phi2Prime = odom_pose[2] - odom_pose_prev[2] - phi1Prime

    #for every particle in the paricle cloud 
    for m in range(M):

        #assuming local pose change in small enough to equal global pose change with gaussian process noise 
        phi1[m] = phi1Prime + randn() * gauss_std 
        phi2[m] = phi2Prime + randn() * gauss_std 
        d[m] = dPrime + randn()* 4* gauss_std 

        #updating particle poses in x y theta 
        particle_poses[m, 0] += d[m]*cos(particle_poses[m, 2] + phi1[m]) 
        particle_poses[m, 1] += d[m]*sin(particle_poses[m, 2] + phi1[m]) 
        particle_poses[m, 2] += phi1[m] + phi2[m]

    return particle_poses

#Particle filter sensor model - weighting of certainty for each particle
def sensor_model(particle_poses, beacon_pose, beacon_loc):

    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)
    r_particle = np.zeros(M)
    phi_particle = np.zeros(M)
    r_error = np.zeros(M)
    phi_error = np.zeros(M)
    r_erroravg = np.zeros(M)

    #range and bearing standard deviations - trial and error for tweak weighting
    r_std = 0.05
    phi_std = 0.05 
    mu = 0
    

    #range and bearing w.r.t robot given est beacon poses
    r = np.sqrt(beacon_pose[0]**2 + beacon_pose[1]**2) 
    phi = np.arctan2(beacon_pose[1],beacon_pose[0])


    for m in range(M-1):
        #range and bearing of each particle with relatve to bearing position
        r_particle[m] = np.sqrt((beacon_loc[0]-particle_poses[m,0])**2 + (beacon_loc[1]-particle_poses[m,1])**2) 
        phi_particle[m] = angle_difference(particle_poses[m,2],np.arctan2((beacon_loc[1]-particle_poses[m,1]),(beacon_loc[0]-particle_poses[m,0]))) 

        #range and bearing error for PDF 
        r_error[m] = r - r_particle[m] 
        phi_error[m] = angle_difference(phi_particle[m],phi)

        #Multiplicaiton of independent gaussian PDFs for range and bearing 
        particle_weights[m] = gauss(r_error[m],mu,r_std)*gauss(phi_error[m],mu,phi_std)
        r_erroravg[m] = np.mean(r_error)
    
    #print(np.mean(r_erroravg))
    return particle_weights

 
    
 
