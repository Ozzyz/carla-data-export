"""
This file contains all the methods responsible for saving the generated data in the correct output format.

"""
import cv2
import numpy as np
import os
import logging

from utils import degrees_to_radians


def save_groundplanes(planes_fname, player_measurements, lidar_height):
    from math import cos, sin
    """ Saves the groundplane vector of the current frame. 
        The format of the ground plane file is first three lines describing the file (number of parameters).
        The next line is the three parameters of the normal vector, and the last is the height of the normal vector,
        which is the same as the distance to the camera in meters.
    """
    rotation = player_measurements.transform.rotation
    pitch, roll = rotation.pitch, rotation.roll
    # Since measurements are in degrees, convert to radians
    pitch = degrees_to_radians(pitch)
    roll = degrees_to_radians(roll)
    # Rotate normal vector (y) wrt. pitch and yaw
    normal_vector = [cos(pitch)*sin(roll), 
                     cos(pitch)*cos(roll), 
                     sin(pitch)
                    ]
    normal_vector = map(str, normal_vector)
    with open(planes_fname, 'w') as f:
        f.write("# Plane\n")
        f.write("Width 4\n")
        f.write("Plane 1\n")
        f.write("{} {}\n".format(" ".join(normal_vector), lidar_height))
    logging.info("Wrote plane data to %s", planes_fname)


def save_ref_files(OUTPUT_FOLDER, id):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt']:
        path = os.path.join(OUTPUT_FOLDER, name)
        with open(path, 'a') as f:
            f.write(id + '\n')
        logging.info("Wrote reference files to %s", path)

def save_image_data(filename, image):
    logging.info("Wrote image data to %s", filename)
    # Convert to correct color format
    color_fmt = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, color_fmt)

def save_lidar_data(filename, lidar_measurement, format="bin"):
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
    """
    logging.info("Wrote lidar data to %s", filename)
    if format == "bin":
        lidar_array = [[point.x, -point.z, -point.y, 1.0] for point in lidar_measurement.point_cloud]  # Hopefully correct format
        lidar_array = np.array(lidar_array).astype(np.float32)
        lidar_array.tofile(filename)
    else:
        lidar_measurement.point_cloud.save_to_disk(filename)

def save_kitti_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti data to %s", filename)

def save_calibration_matrices(filename, intrinsic_mat, extrinsic_mat):
    """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain: 
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
    """
    ravel_mode = 'C'
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)
    R0 = np.identity(3) # NOTE! This assumes that the camera and lidar occupy the same position on the car!!
    TR_velodyne = np.identity(3)
    TR_velodyne= np.column_stack((TR_velodyne, np.array([0, 0, 1])))
    Tr_imu_to_velo = np.identity(3)
    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(map(str, arr.flatten(ravel_mode).squeeze()))))

    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
        for i in range(4): # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "Tr_imu_to_velo", Tr_imu_to_velo)
    logging.info("Wrote all calibration matrices to %s", filename)