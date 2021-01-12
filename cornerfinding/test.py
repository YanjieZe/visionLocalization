import numpy as np

point_world1 = np.array([0,0,0])
point_world2 = np.array([16,0,0])
point_world3 = np.array([0,21,0])
point_world4 = np.array([16,21,0])

point_world = np.stack([point_world1,point_world2,point_world3,point_world4])
print(point_world.shape)