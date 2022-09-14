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
    
    # TODO.  For each particle calculate its predicted pose plus some
    # additive error to represent the process noise.  With this demo
    # code, the particles move in the -y direction with some Gaussian
    # additive noise in the x direction.  Hint, to start with do not
    # add much noise.

    phi1Prime = np.arctan2((odom_pose[1]-odom_pose_prev[1]),(odom_pose[0]-odom_pose_prev[0])) - odom_pose_prev[2]
    dPrime = np.sqrt((odom_pose[1]-odom_pose_prev[1])**2 + (odom_pose[0]-odom_pose_prev[0])**2)
    phi2Prime = odom_pose[2] - odom_pose_prev[2] - phi1Prime

    for m in range(M):
        #particle_poses[m, 0] += dPrime*cos(particle_poses[m, 2] + phi1Prime) + randn(1) * 0.1
        #particle_poses[m, 1] += dPrime*sin(particle_poses[m, 2] + phi1Prime) + randn(1) * 0.1
        #particle_poses[m, 2] += phi1Prime + phi2Prime + randn(1) * 0.1
        # x(m)n = x(m)n-1 + g(x(m)n-1, un-1) + w(m)n

        # # Add gaussian additive noise (randn) in x direction
        # particle_poses[m, 0] += randn(1) * 0.1
        # # Particles move in the -y direction  X AND Y OPPOSITE
        # particle_poses[m, 1] -= 0.1    


        particle_poses[m, 0] += dPrime*cos(particle_poses[m, 2] + phi1Prime) # + randn(1) * 0.1
        particle_poses[m, 1] += dPrime*sin(particle_poses[m, 2] + phi1Prime) # + randn(1) * 0.1
        particle_poses[m, 2] += phi1Prime + phi2Prime

        # x(m)n = x(m)n-1 + g(x(m)n-1, un-1) + w(m)n

        # Add gaussian additive noise in x direction
        particle_poses[m, 0] += randn(1) * 0.01
        particle_poses[m, 1] += randn(1) * 0.05

        # Particles move in the -y direction   WHY IS THE GRAPH X AND Y OPPOSITE TO WHAT YOUD EXPECT FOR A MAP!!!!
        #particle_poses[m, 1] -= 0.1

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
    

    
    # TODO.  For each particle calculate its weight based on its pose,
    # the relative beacon pose, and the beacon location.

    #Implementing equations from multivariate particle filter notes (pg 115-116)


    r_std = 0.1
    phi_std = 0.1
    #calculate range and bearing w.r.t robot given est beacon poses
    r = np.sqrt(beacon_pose[0]**2 + beacon_pose[1]**2) #+ random.gauss(0,r_std)
    phi = np.arctan2(beacon_pose[1],beacon_pose[0]) #+ random.gauss(0,phi_std)

    r_particle = np.zeros(M)
    phi_particle = np.zeros(M)

    r_error = np.zeros(M)
    phi_error = np.zeros(M)

    r_var = np.zeros(M)
    phi_var = np.zeros(M)


   
    #for m in range(M-1):
     #   particle_weights[m] = 1
    #return particle_weights



    
    for m in range(M-1):
        r_particle[m] = np.sqrt((beacon_loc[0]-particle_poses[m,0])**2 + (beacon_loc[1]-particle_poses[m,1])**2) 
        phi_particle[m] = angle_difference(particle_poses[m,2],np.arctan2((beacon_loc[1]-particle_poses[m,1]),(beacon_loc[0]-particle_poses[m,0]))) 
        r_error[m] = r - r_particle[m] 
        #print("range error = {}". format(r_error[m]))
        phi_error[m] = angle_difference(phi_particle[m],phi)
        #print("bearing error = {}". format(phi_error[m]))
        r_var[m] = np.var(r_particle[m])
        phi_var[m] = np.var(phi_particle[m])
        #particle_weights[m] = gauss(r_error[m],r_var[m],r_std)*gauss(phi_error[m],phi_var[m],phi_std)# r and phi errors need to be probability density functions, how do you implenet that in python?
        particle_weights[m] = gauss(r_error[m],0,r_std)*gauss(phi_error[m],0,phi_std)
        #print(" particle_weights = {}".format(particle_weights[m]))
    return particle_weights

 
    
 
