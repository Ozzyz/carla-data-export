#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

import argparse
import cv2
import logging
import random
import time
import math
import colorsys
import os
try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
    from numpy.linalg import pinv, inv
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.transform import Transform


from utils import Timer, rand_color, vector3d_to_list, degrees_to_radians
from datadescriptor import KittiDescriptor
from lidar_utils import *
from camera_utils import *
from dataexport import *

""" DATA GENERATION SETTINGS"""
GEN_DATA = True # Whether or not to save training data
STEPS_BETWEEN_RECORDINGS = 10 # How many frames to wait between each capture of screen, bounding boxes and lidar
CLASSES_TO_LABEL = ["Vehicle"] #, "Pedestrian"]
LIDAR_DATA_FORMAT = "bin" # Lidar can be saved in bin to comply to kitti, or the standard .ply format
assert LIDAR_DATA_FORMAT in ["bin", "ply"], "Lidar data format must be either bin or ply"
OCCLUDED_VERTEX_COLOR = (255, 0, 0)
VISIBLE_VERTEX_COLOR = (0, 255, 0)


""" CARLA SETTINGS """
CAMERA_HEIGHT_POS = 1.8
MIN_BBOX_AREA_IN_PX = 100
LIDAR_HEIGHT_POS = CAMERA_HEIGHT_POS

""" AGENT SETTINGS """
NUM_VEHICLES = 20
NUM_PEDESTRIANS = 10

""" RENDERING SETTINGS """
WINDOW_WIDTH = 1248
WINDOW_HEIGHT = 384
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

WINDOW_WIDTH_HALF = WINDOW_WIDTH / 2
WINDOW_HEIGHT_HALF = WINDOW_HEIGHT / 2

MAX_RENDER_DEPTH_IN_METERS = 70 # Meters 
MIN_VISIBLE_VERTICES_FOR_RENDER = 4

""" OUTPUT FOLDER GENERATION """
PHASE = "training"
OUTPUT_FOLDER = os.path.join("_out", PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne', 'planes']

def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)

for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)


def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(0, 0.0, CAMERA_HEIGHT_POS)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)

    lidar = sensor.Lidar('Lidar32')
    lidar.set_position(0, 0.0, LIDAR_HEIGHT_POS)
    lidar.set_rotation(0, 0, 0)
    lidar.set(
        Channels=40,
        Range=MAX_RENDER_DEPTH_IN_METERS,
        PointsPerSecond=720000,
        RotationFrequency=20,
        UpperFovLimit=7,
        LowerFovLimit=-16)    
    settings.add_sensor(lidar)
    """ Depth camera for filtering out occluded vehicles """
    depth_camera = sensor.Camera('DepthCamera', PostProcessing='Depth')
    depth_camera.set(FOV=90.0)
    depth_camera.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    depth_camera.set_position(0, 0, CAMERA_HEIGHT_POS)
    depth_camera.set_rotation(0, 0, 0)
    settings.add_sensor(depth_camera)
    # (Intrinsic) K Matrix
    # | f 0 Cu
    # | 0 f Cv
    # | 0 0 1
    # (Cu, Cv) is center of image
    k = np.identity(3)
    k[0, 2] = WINDOW_WIDTH_HALF
    k[1, 2] = WINDOW_HEIGHT_HALF
    f = WINDOW_WIDTH / \
        (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    camera_to_car_transform = camera0.get_unreal_transform()
    return settings, k, camera_to_car_transform


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings, self._intrinsic, self._camera_to_car_transform = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 16.43, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
        self.captured_frame_no = 0
        self._measurements = None
        self._extrinsic = None

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        logging.info('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        # (Extrinsic) Rt Matrix
        # (Camera) local 3d to world 3d.
        # Get the transform from the player protobuf transformation.
        world_transform = Transform(
            measurements.player_measurements.transform
        )
        # Compute the final transformation matrix.
        self._extrinsic = world_transform * self._camera_to_car_transform
        self._measurements = measurements
        self._main_image = sensor_data.get('CameraRGB', None)
        self._lidar_measurement = sensor_data.get('Lidar32', None)
        self._depth_image = sensor_data.get('DepthCamera', None)

        # Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)
            else:
                self._print_player_measurements(measurements.player_measurements)

            # Plot position on the map as well.
            self._timer.lap()

        control = self._get_keyboard_control(pygame.key.get_pressed())
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        control.reverse = self._is_on_reverse
        return control

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):
        all_datapoints = []
        save_data_now = True
 
        if self._main_image is not None and self._depth_image is not None:
            depth_map = to_depth_array(self._depth_image, self._intrinsic)
            array = image_converter.to_rgb_array(self._main_image)
            array = array.copy() # array.setflags(write=1)
            # Stores all datapoints for the current frames
            for agent in self._measurements.non_player_agents:
                if should_detect_class(agent):
                    array, kitti_datapoint = bbox_from_agent(agent, self._intrinsic, self._extrinsic.matrix, array, depth_map)
                    if kitti_datapoint:
                        rotation_y = self.get_relative_rotation_y(agent) % math.pi
                        kitti_datapoint.set_rotation_y(rotation_y)
                        all_datapoints.append(kitti_datapoint)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))
        else:
            save_data_now = False
        """
        if self._lidar_measurement is not None:
            #if self._main_image is not None:
                #array = image_converter.to_rgb_array(self._main_image)
                #array = array.copy()
                #lidar_world_pos = np.add(vector3d_to_list(self._measurements.player_measurements.transform.location), [0, 0.0, 1.8])
                #array = project_point_cloud(array, self._lidar_measurement.data, lidar_world_pos , self._extrinsic.matrix, self._intrinsic)
                #surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                #self._display.blit(surface, (0, 0))
            #logging.info("Shape of lidar data: ", self._lidar_measurement.data.shape)
            lidar_data = np.array(self._lidar_measurement.data[:, :2])
            lidar_data *= 2.0
            lidar_data += 100.0
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            #draw lidar
            lidar_img_size = (200, 200, 3)
            lidar_img = np.zeros(lidar_img_size)
            print(lidar_data.shape)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            surface = pygame.surfarray.make_surface(lidar_img)
            self._display.blit(surface, (10, 10))
        else:
            logging.info("Lidar data is None!")
            save_data_now = False
        """
        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] * (new_window_width/float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])

                    w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
                    h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))
        # Save screen, lidar and kitti training labels
        if self._timer.step % STEPS_BETWEEN_RECORDINGS == 0:
            if save_data_now and GEN_DATA and all_datapoints:
                logging.info("Attempting to save at timer step {}, frame no: {}".format(self._timer.step, self.captured_frame_no))
                groundplane_fname = os.path.join(OUTPUT_FOLDER, 'planes/{0:06}.txt'.format(self.captured_frame_no))
                lidar_fname = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin'.format(self.captured_frame_no))
                kitti_fname = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt'.format(self.captured_frame_no))
                img_fname = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png'.format(self.captured_frame_no))
                calib_filename =  os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt'.format(self.captured_frame_no))
                save_groundplanes(groundplane_fname, self._measurements.player_measurements, LIDAR_HEIGHT_POS)
                save_ref_files(OUTPUT_FOLDER, "{0:06}".format(self.captured_frame_no))
                save_image_data(img_fname, image_converter.to_rgb_array(self._main_image))
                save_kitti_data(kitti_fname, all_datapoints)
                save_lidar_data(lidar_fname, self._lidar_measurement, LIDAR_DATA_FORMAT)
                save_calibration_matrices(calib_filename, self._intrinsic, self._extrinsic)
                self.captured_frame_no += 1
            else:
                logging.info("ould not save training data - no visible agents in scene")

        pygame.display.flip()

    def get_relative_rotation_y(self, agent):
        """ Returns the relative rotation of the agent to the camera in yaw
        The relative rotation is the difference between the camera rotation (on car) and the agent rotation"""
        # We only car about the rotation for the classes we do detection on
        if agent.vehicle.transform:
            rot_agent = agent.vehicle.transform.rotation.yaw
            rot_car = self._measurements.player_measurements.transform.rotation.yaw
            return degrees_to_radians(rot_agent - rot_car)


def to_depth_array(depth_image, k):
    """ Converts a raw depth image from Camera depth sensor to an array where each index 
        is the depth value. 
        This conversion is needed because the depth camera encodes depth in the RGB-values
        as d = (R + G*256 + B*256*256)/(256*256*256 - 1) * FAR_DISTANCE
        K is the intrinsic matrix
    """
    from numpy.matlib import repmat
    far_distance_in_meters = 1000
    # RGB image will have shape (WINDOW_HEIGHT, WINDOW_WIDTH, 3)
    array = image_converter.to_bgra_array(depth_image)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth = normalized_depth * far_distance_in_meters
    return depth
    

def should_detect_class(agent):
    """ Returns true if the agent is of the classes that we want to detect.
        Note that Carla has class types in lowercase 
    """
    return True in [agent.HasField(class_type.lower()) for class_type in CLASSES_TO_LABEL]



def transforms_from_agent(agent):
    """ Returns the KITTI object type and transforms, locations and extension of the given agent """
    if agent.HasField('pedestrian'):
        obj_type = 'Pedestrian'
        agent_transform = Transform(agent.pedestrian.transform)
        bbox_transform = Transform(agent.pedestrian.bounding_box.transform)
        ext = agent.pedestrian.bounding_box.extent
        location = agent.pedestrian.transform.location
    elif agent.HasField('vehicle'):
        obj_type = 'Car'
        agent_transform = Transform(agent.vehicle.transform)
        bbox_transform = Transform(agent.vehicle.bounding_box.transform)
        ext = agent.vehicle.bounding_box.extent
        location = agent.vehicle.transform.location
    else:
        return (None, None, None, None, None)
    return obj_type, agent_transform, bbox_transform, ext, location

def vertices_from_extension(ext):
    """ Extraxts the 8 bounding box vertices relative to (0,0,0)
    https://github.com/carla-simulator/carla/commits/master/Docs/img/vehicle_bounding_box.png 
    8 bounding box vertices relative to (0,0,0)
    """
    return np.array([
        [  ext.x,   ext.y,   ext.z], # Top left front
        [- ext.x,   ext.y,   ext.z], # Top left back
        [  ext.x, - ext.y,   ext.z], # Top right front
        [- ext.x, - ext.y,   ext.z], # Top right back
        [  ext.x,   ext.y, - ext.z], # Bottom left front
        [- ext.x,   ext.y, - ext.z], # Bottom left back
        [  ext.x, - ext.y, - ext.z], # Bottom right front
        [- ext.x, - ext.y, - ext.z]  # Bottom right back
    ])

def bbox_from_agent(agent, intrinsic_mat, extrinsic_mat, array, depth_map):
    """ Creates bounding boxes for a given agent and camera/world calibration matrices.
        Returns the modified array that contains the screen rendering with drawn on vertices from the agent """
    # get the needed transformations
    # remember to explicitly make it Transform() so you can use transform_points()
    obj_type, agent_transform, bbox_transform, ext, location = transforms_from_agent(agent)
    if obj_type is None:
        logging.warning("Could not get bounding box for agent. Valid classes : %s", CLASSES_TO_LABEL)
        return array, []
    
    bbox = vertices_from_extension(ext)
    # Shows which verticies that are connected so that we can draw lines between them
    # The key of the dictionary is the index in the bbox array, and the corresponding value is a list of indices 
    # referring to the same array.
    vertex_graph = {0: [1, 2, 4], 
                    1: [0, 3, 5],
                    2: [0, 3, 6], 
                    3: [1, 2, 7], 
                    4: [0, 5, 6], 
                    5: [1, 4, 7], 
                    6: [2,4,7]}

    # transform the vertices respect to the bounding box transform
    bbox = bbox_transform.transform_points(bbox)

    # the bounding box transform is respect to the agents transform
    # so let's transform the points relative to it's transform
    bbox = agent_transform.transform_points(bbox)

    # agents's transform is relative to the world, so now,
    # bbox contains the 3D bounding box vertices relative to the world
    # Additionally, you can logging.info these vertices to check that is working
    # Store each vertex 2d points for drawing bounding boxes later
    vertices_pos2d = []
    num_visible_vertices = 0
    num_vertices_outside_camera = 0
    for vertex in bbox:
        # World coordinates
        pos_vector = np.array([
            [vertex[0,0]],  # [[X,
            [vertex[0,1]],  #   Y,
            [vertex[0,2]],  #   Z,
            [1.0]           #   1.0]]
        ])
        # Camera coordinates
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        # 2d pixel coordinates
        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)
        
        vertex_depth = pos2d[2] # The actual rendered depth (may be wall or other object instead of vertex)
        x_2d, y_2d  = WINDOW_WIDTH - pos2d[0],  WINDOW_HEIGHT - pos2d[1]
        vertices_pos2d.append((y_2d, x_2d))

        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((y_2d, x_2d)): # if the point is in front of the camera but not too far away
            is_occluded = point_is_occluded((y_2d, x_2d), vertex_depth, depth_map)
            if is_occluded:
                vertex_color = OCCLUDED_VERTEX_COLOR
            else:
                num_visible_vertices += 1
                vertex_color = VISIBLE_VERTEX_COLOR
            draw_rect(array, (y_2d, x_2d), 4, vertex_color)
        else:
            num_vertices_outside_camera += 1

    midpoint = midpoint_from_agent_location(array, location, extrinsic_mat, intrinsic_mat)

    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < MIN_VISIBLE_VERTICES_FOR_RENDER: # At least N vertices has to be visible in order to draw bbox
        bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
        area = calc_bbox2d_area(bbox_2d)
        if area < MIN_BBOX_AREA_IN_PX:
            logging.info("Filtered out bbox with too low area {}".format(area))
            return array, None
        datapoint = KittiDescriptor()
        datapoint.set_type(obj_type)
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(ext)
        datapoint.set_3b_object_location(midpoint)
        draw_3d_bounding_box(array, vertices_pos2d, vertex_graph)
        return array, datapoint
    else:
        return array, None



def calc_bbox2d_area(bbox_2d):
    """ Calculate the area of the given 2d bbox
    Input is assumed to be xmin, ymin, xmax, ymax tuple 
    """
    xmin, ymin, xmax, ymax = bbox_2d
    return (ymax - ymin) * (xmax - xmin)


def parse_args():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='logging.info debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()
    return args

def main():
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    logging.info(__doc__)

    while True:
        try:
            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info('\nCancelled by user. Bye!')
