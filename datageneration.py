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
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
    from numpy.linalg import pinv, inv
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.tcp import TCPConnectionError
from carla.transform import Transform, Scale


from utils import Timer, rand_color, vector3d_to_list, degrees_to_radians
from datadescriptor import KittiDescriptor
from dataexport import *
from bounding_box import create_kitti_datapoint
from carla_utils import KeyboardHelper, MeasurementsDisplayHelper
from constants import *
from settings import make_carla_settings

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

""" DATA SAVE PATHS """
GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, 'planes/{0:06}.txt')
LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')
CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings, self._intrinsic, self._camera_to_car_transform, self._lidar_to_car_transform = make_carla_settings(args)
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

                MeasurementsDisplayHelper.print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation, self._timer)
            else:
                MeasurementsDisplayHelper.print_player_measurements(measurements.player_measurements, self._timer)

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
        control = KeyboardHelper.get_keyboard_control(keys, self._is_on_reverse, self._enable_autopilot)
        if control is not None:
            control, self._is_on_reverse, self._enable_autopilot = control
        return control


    def _on_render(self):
        datapoints = []
        if self._main_image is not None and self._depth_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            array = array.copy()
            # Stores all datapoints for the current frames
            for agent in self._measurements.non_player_agents:
                if should_detect_class(agent):
                    array, kitti_datapoint = create_kitti_datapoint(agent, self._intrinsic, self._extrinsic.matrix, array, self._depth_image, self._measurements.player_measurements)
                    if kitti_datapoint:
                        datapoints.append(kitti_datapoint)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))
        
        if self._map_view is not None:
            self._display_agents(self._map_view)
           
        # Save screen, lidar and kitti training labels together with calibration and groundplane files
        if self._timer.step % STEPS_BETWEEN_RECORDINGS == 0:
            if GEN_DATA and datapoints:
                self._save_training_files(datapoints)
                self.captured_frame_no += 1
            else:
                logging.info("Could not save training data - no visible agents of selected classes in scene")

        pygame.display.flip()

    def _save_training_files(self, datapoints):
        logging.info("Attempting to save at timer step {}, frame no: {}".format(self._timer.step, self.captured_frame_no))
        groundplane_fname = GROUNDPLANE_PATH.format(self.captured_frame_no)
        lidar_fname = LIDAR_PATH.format(self.captured_frame_no)
        kitti_fname = LABEL_PATH.format(self.captured_frame_no)
        img_fname = IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = CALIBRATION_PATH.format(self.captured_frame_no)

        save_groundplanes(groundplane_fname, self._measurements.player_measurements, LIDAR_HEIGHT_POS)
        save_ref_files(OUTPUT_FOLDER, self.captured_frame_no)
        save_image_data(img_fname, image_converter.to_rgb_array(self._main_image))
        save_kitti_data(kitti_fname, datapoints)
        save_lidar_data(lidar_fname, self._lidar_measurement, self._lidar_to_car_transform, LIDAR_HEIGHT_POS, LIDAR_DATA_FORMAT)
        save_calibration_matrices(calib_filename, self._intrinsic, self._extrinsic)

    def _display_agents(self, map_view):
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


def should_detect_class(agent):
    """ Returns true if the agent is of the classes that we want to detect.
        Note that Carla has class types in lowercase 
    """
    return True in [agent.HasField(class_type.lower()) for class_type in CLASSES_TO_LABEL]


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
