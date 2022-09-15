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


def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt):
    """Apply motion model and return updated array of particle_poses.

    Parameters
    ----------

    particle_poses: an M x 3 array of particle_poses where M is the
    number of particles.  Each pose is (x, y, theta) where x and y are
    in metres and theta is in radians.

    speed_command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    dt is the time step (s).

    Returns
    -------
    An M x 3 array of updated particle_poses.

    """

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

        #assuming local pose change in small enough to eqal global pose change with gaussian proccess noise 
        phi1[m] = phi1Prime + randn() * 0.001 # 0.001
        phi2[m] = phi2Prime + randn() * 0.001 #0.001
        d[m] = dPrime + randn()* 0.004 # 0.0035

    #converting global pose change back to coordinates x,y and theta 
        particle_poses[m, 0] += d[m]*cos(particle_poses[m, 2] + phi1[m]) 
        particle_poses[m, 1] += d[m]*sin(particle_poses[m, 2] + phi1[m]) 
        particle_poses[m, 2] += phi1[m] + phi2[m]

    return particle_poses

def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.
    Parameters
    ----------
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system. [rxb]

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system. [mxb]

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """
    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)
    
    #range and bearing w.r.t robot given est beacon poses
    r = np.sqrt(beacon_pose[0]**2 + beacon_pose[1]**2) 
    phi = np.arctan2(beacon_pose[1],beacon_pose[0])

    r_particle = np.zeros(M)
    phi_particle = np.zeros(M)

    r_error = np.zeros(M)
    phi_error = np.zeros(M)

    #range and bearing standard deviations - trial and error for tweak weighting
    r_std = 0.05
    phi_std = 0.04 
    mu = 0


    for m in range(M-1):
        #range and bearing of each particle with relatve to bearing position
        r_particle[m] = np.sqrt((beacon_loc[0]-particle_poses[m,0])**2 + (beacon_loc[1]-particle_poses[m,1])**2) 
        phi_particle[m] = angle_difference(particle_poses[m,2],np.arctan2((beacon_loc[1]-particle_poses[m,1]),(beacon_loc[0]-particle_poses[m,0]))) 

        #range and bearing error for PDF 
        r_error[m] = r - r_particle[m] 
        phi_error[m] = angle_difference(phi_particle[m],phi)

        #Multiplicaiton of independent gaussian PDFs for range and bearing 
        particle_weights[m] = gauss(r_error[m],mu,r_std)*gauss(phi_error[m],mu,phi_std)
    
    return particle_weights

 
    
 
