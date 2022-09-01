"""Particle filter sensor and motion model implementations.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy.random import randn
from utils import gauss, wraptopi, angle_difference


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
        particle_poses[m, 0] += dPrime*cos(particle_poses[m, 2] + phi1Prime) + randn(1) * 0.1
        particle_poses[m, 1] += dPrime*sin(particle_poses[m, 2] + phi1Prime) + randn(1) * 0.1
        particle_poses[m, 2] += phi1Prime + phi2Prime + randn(1) * 0.1
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
        particle_poses[m, 0] += randn(1) * 0.1
        # Particles move in the -y direction   WHY IS THE GRAPH X AND Y OPPOSITE TO WHAT YOUD EXPECT FOR A MAP!!!!
        particle_poses[m, 1] -= 0.1

    return particle_poses

def angdiff(theta1, theta2):
    return min(2*np.pi-(theta1-theta2),theta1-theta2)

def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system.

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system.

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
    r = np.sqrt(beacon_pose[0]**2 + beacon_pose[1]**2)
    phi = np.arctan2(beacon_pose[1],beacon_pose[0])

    for m in range(M):
        r_particle[m] = np.sqrt((beacon_loc[0]-beacon_pose[m,0])**2 + (beacon_loc[1]-beacon_pose[1]**2))
        phi_particle[m] = angdiff(np.arctan2(beacon_loc[1]-beacon_pose[1],beacon_loc[0]-beacon_pose[m,0]),beacon_pose[2])
        r_error = r-r_particle[m]
        phi_error = phi - phi_particle[m]
        particle_weights[m] = r_error*phi_error # r and phi errors need to be probability density functions, how do you implenet that in python?

    return particle_weights
