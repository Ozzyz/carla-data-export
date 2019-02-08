

from camera_utils import proj_to_camera, proj_to_2d, draw_rect
from datageneration import WINDOW_WIDTH, WINDOW_HEIGHT
import numpy as np


def project_point_cloud(array, point_cloud, sensor_world_pos, extrinsic_mat, intrinsic_mat, draw_each_nth=10):
    """ Projects the lidar measurements onto the screen and draws them.
    Since the points are just X,Y,Z coordinates relative to the lidar, this can be done by a simple projection. 
    Note that this assumes that the camera and the lidar have exactly the same position and rotation!!
    """
    num_samples, dim = point_cloud.shape
    assert dim == 3, "Point cloud should have shape (?, 3) (X, Y, Z)"
    # World vectors have x,y,z,1
    world_vecs = np.zeros(shape=(dim+1, num_samples))
    for i in range(0, num_samples, draw_each_nth):
        # TODO: Find out if this is legal, since they may not have same coordinate system directions
        # Since the points are relative to sensors, add the sensor world position to get world coordinates
        vec = point_cloud[i, :]
        world_3d_vec = vec + np.array(sensor_world_pos)
        #print("World 3d vec: ", world_3d_vec)
        pos_vector = np.array([
            [world_3d_vec[0]],  # [[X,
            [world_3d_vec[1]],  #   Y,
            [world_3d_vec[2]],  #   Z,
            [1.0]           #   1.0]]
        ])
        world_vecs[:, i] = pos_vector.ravel()
        #print("Transformed lidar pos ", pos_vector)
    
    camera_projected_vecs = proj_to_camera(world_vecs, extrinsic_mat)
    pos2d = np.dot(intrinsic_mat, camera_projected_vecs[:3])
    #print("Shape of pos2d: ", pos2d.shape)
    for j in range(0, num_samples, draw_each_nth):
        cur_pos2d = pos2d[..., j]
        cur_pos2d = np.array([
            cur_pos2d[0] / cur_pos2d[2],
            cur_pos2d[1] / cur_pos2d[2],
            cur_pos2d[2]
        ])
        depth = cur_pos2d[2]
        
        if 100 > depth > 0:
            x_2d = WINDOW_WIDTH - cur_pos2d[0]
            y_2d = WINDOW_HEIGHT - cur_pos2d[1]
            draw_rect(array, (y_2d, x_2d), int(10/depth))
            
    return array
