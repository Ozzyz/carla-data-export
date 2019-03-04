""" DATA GENERATION SETTINGS"""
GEN_DATA = True # Whether or not to save training data
STEPS_BETWEEN_RECORDINGS = 10 # How many frames to wait between each capture of screen, bounding boxes and lidar
CLASSES_TO_LABEL = ["Vehicle"] #, "Pedestrian"]
LIDAR_DATA_FORMAT = "bin" # Lidar can be saved in bin to comply to kitti, or the standard .ply format
assert LIDAR_DATA_FORMAT in ["bin", "ply"], "Lidar data format must be either bin or ply"
OCCLUDED_VERTEX_COLOR = (255, 0, 0)
VISIBLE_VERTEX_COLOR = (0, 255, 0)


""" CARLA SETTINGS """
CAMERA_HEIGHT_POS = 1.6
LIDAR_HEIGHT_POS = CAMERA_HEIGHT_POS
MIN_BBOX_AREA_IN_PX = 100


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