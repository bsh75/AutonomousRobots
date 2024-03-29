"""Particle filter demonstration program.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury

Note, this example is written in an attempt to be straightforward
to follow for those new to Python.  Normally, we would have written
it using about a dozen classes.
"""

from __future__ import print_function, division
from numpy.random import uniform, seed
try:
    import matplotlib; matplotlib.use("TkAgg")        
except:
    import matplotlib; matplotlib.use("Qt5Agg")

from models import motion_model, sensor_model
from utils import *
from plot import *
from transform import *
import numpy as np


seed(1)

# Load data

# data is a (many x 13) matrix. Its columns are:
# time_ns, velocity_command, rotation_command, map_x, map_y, map_theta, odom_x, odom_y, odom_theta,
# beacon_ids, beacon_x, beacon_y, beacon_theta
data = np.genfromtxt('assignment1/partB/data.csv', delimiter=',', skip_header=1)

# Time in ns
t = data[:, 0]

# Velocity command in m/s, rotation command in rad/sqq
commands = data[:, 1:3]

# Position in map frame, from SLAM (this approximates ground truth)
slam_poses = data[:, 3:6]

# Position in odometry frame, from wheel encoders and gyro
odom_poses = data[:, 6:9]

# Id and measured position of beacon in camera frame
beacon_ids = data[:, 9]
beacon_poses = data[:, 10:13]
# Use beacon id of -1 if no beacon detected
beacon_ids[np.isnan(beacon_ids)] = -1
beacon_ids = beacon_ids.astype(int)
beacon_visible = beacon_ids >= 0

# map_data is a 16x13 matrix.  Its columns are:
# beacon_ids, x, y, theta, (9 columns of covariance)
map_data = np.genfromtxt('assignment1/partB/beacon_map.csv', delimiter=',', skip_header=1)

Nbeacons = map_data.shape[0]
beacon_locs = np.zeros((Nbeacons, 3))
for m in range(Nbeacons):
    id = int(map_data[m, 0])
    beacon_locs[id] = map_data[m, 1:4]

# Remove jumps in the pose history
slam_poses = clean_poses(slam_poses)

# Transform odometry poses into map frame
odom_to_map = find_transform(odom_poses[0], slam_poses[0]) # Return the transformation from `odom_poses[0]` to `slam_poses[0]`
odom_poses = transform_pose(odom_to_map, odom_poses) # Apply transformation 'odom_to_map' (dx, dy, dtheta) to 'odom_pose' (x, y, theta)

plt.ion()
fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(111)
fig.canvas.mpl_connect('key_press_event', keypress_handler)
fig.canvas.manager.full_screen_toggle()

plot_beacons(axes, beacon_locs, label='Beacons')
plot_path(axes, slam_poses, '-', label='SLAM')
#plot_path(axes, slam_poses, '-', label='SLAM')
# Uncomment to show odometry when debugging
#plot_path(axes, odom_poses, 'b:', label='Odom')

axes.legend(loc='lower right')

axes.set_xlim([-6, None])
axes.axis('equal')

# Tweak axes to make plotting better
axes.invert_yaxis()
axes.set_xlabel('y')
axes.set_ylabel('x')
axes.figure.canvas.draw()
axes.figure.canvas.flush_events()


start_step = 0

#number of particles in filter
Nparticles = 100

#  How many steps between display updates
display_steps = 100

#Motion PDF standard devation and std if particle weights go below threshold
PDF_std = 0.001
std_lost = 10* PDF_std

#  Set initial belief.  This assumes a uniform distribution for the pose
# around the known starting pose. 
start_pose = slam_poses[start_step]
Xmin = start_pose[0] - 0.3
Xmax = start_pose[0] + 0.3
Ymin = start_pose[1] - 0.3
Ymax = start_pose[1] + 0.3
Tmin = start_pose[2] - 0.3
Tmax = start_pose[2] + 0.3

weights = np.ones(Nparticles)
poses = np.zeros((Nparticles, 3))

for m in range(Nparticles):
    poses[m] = (uniform(Xmin, Xmax),
                uniform(Ymin, Ymax),
                uniform(Tmin, Tmax))

Nposes = odom_poses.shape[0]
est_poses = np.zeros((Nposes, 3))

plot_particles(axes, poses, weights)
axes.set_title('Push space to start/stop, dot to move one step, q to quit...')
wait_until_key_pressed()

state = 'run'
display_step_prev = 0
for n in range(start_step + 1, Nposes):

   
    poses = motion_model(poses, commands[n-1], odom_poses[n], odom_poses[n - 1],
                         t[n] - t[n - 1], PDF_std)
    
    if beacon_visible[n]:

        beacon_id = beacon_ids[n]
        beacon_loc = beacon_locs[beacon_id]
        beacon_pose = beacon_poses[n]

       
        weights *= sensor_model(poses, beacon_pose, beacon_loc)

        if sum(weights) < 1e-50:
           print('All weights are close to zero, you are lost...')
            poses = motion_model(poses, commands[n-1], odom_poses[n], odom_poses[n - 1],
                         t[n] - t[n - 1],std_lost)
            

        if is_degenerate(weights):
            print('Resampling %d' % n)
            resample(poses, weights)

    est_poses[n] = poses.mean(axis=0)

    if (n > display_step_prev + display_steps) or state == 'step':
        print(n)

        # Show particle cloud
        plot_particles(axes, poses, weights)

        # Leave breadcrumbs showing current odometry
        plot_path(axes, odom_poses[n], 'k.', label = 'Odometry')

        # Show mean estimate
        plot_path_with_visibility(axes, est_poses[display_step_prev-1 : n+1],
                                  '-', visibility=beacon_visible[display_step_prev-1 : n+1])
        display_step_prev = n

        print(state)

    key = get_key()
    if key == '.':
        state = 'step'
    elif key == ' ':
        if state == 'run':
            state = 'pause'
        else:
            state = 'run'

    if state == 'pause':
        wait_until_key_pressed()
    elif state == 'step':
        wait_until_key_pressed()            


# Display final plot
print('Done, displaying final plot')
plt.ioff()
plt.show()

# Save final plot to file
plot_filename = 'path.pdf'
print('Saving final plot to', plot_filename)

plot_path(axes, est_poses, 'r-', label='PF')
axes.legend(loc='lower right')

fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(111)

plot_beacons(axes, beacon_locs, label='Beacons')
plot_path(axes, slam_poses, 'b-', label='SLAM')
plot_path(axes, odom_poses, 'b:', label='Odom')
plot_path(axes, est_poses, 'r-', label='PF')
axes.legend(loc='lower right')

axes.set_xlim([-6, None])
axes.axis('equal')

# Tweak axes to make plotting better
axes.invert_yaxis()
axes.set_xlabel('y')
axes.set_ylabel('x')
fig.savefig(plot_filename, bbox_inches='tight')
