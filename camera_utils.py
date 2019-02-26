import numpy as np
from numpy.linalg import pinv, inv
from datageneration import WINDOW_HEIGHT, WINDOW_WIDTH

def calc_projected_2d_bbox(vertices_pos2d):
    """ Takes in all vertices in pixel projection and calculates min and max of all x and y coordinates.
        Returns left top, right bottom pixel coordinates for the 2d bounding box as a list of four values.
        Note that vertices_pos2d contains a list of (y_pos2d, x_pos2d) tuples, or None
    """
    
    legal_pos2d = list(filter(lambda x: x is not None, vertices_pos2d))
    y_coords, x_coords = [int(x[0][0]) for x in legal_pos2d], [int(x[1][0]) for x in legal_pos2d]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return [min_x, min_y, max_x, max_y]

def midpoint_from_agent_location(array, location, extrinsic_mat, intrinsic_mat):
    # Calculate the midpoint of the bottom chassis
    # This is used since kitti treats this point as the location of the car
    midpoint_vector = np.array([
        [location.x],  # [[X,
            [location.y],  #   Y,
            [location.z],  #   Z,
            [1.0]           #   1.0]]
    ])
    transformed_3d_midpoint = proj_to_camera(midpoint_vector, extrinsic_mat)
    return transformed_3d_midpoint


def proj_to_camera(pos_vector, extrinsic_mat):
    # transform the points to camera
    #print("Multiplied {} matrix with {} vector".format(extrinsic_mat.shape, pos_vector.shape))
    transformed_3d_pos = np.dot(inv(extrinsic_mat), pos_vector)
    return transformed_3d_pos

def proj_to_2d(camera_pos_vector, intrinsic_mat):
    # transform the points to 2D
        pos2d = np.dot(intrinsic_mat, camera_pos_vector[:3])
        # normalize the 2D points
        pos2d = np.array([
            pos2d[0] / pos2d[2],
            pos2d[1] / pos2d[2],
            pos2d[2]
        ])
        return pos2d

def draw_3d_bounding_box(array, vertices_pos2d, vertex_graph):
    """ Draws lines from each vertex to all connected vertices """
    # Note that this can be sped up by not drawing duplicate lines
    for vertex_idx in vertex_graph:
        neighbour_idxs = vertex_graph[vertex_idx]
        from_pos2d = vertices_pos2d[vertex_idx]
        for neighbour_idx in neighbour_idxs:
            to_pos2d = vertices_pos2d[neighbour_idx]
            if from_pos2d is None or to_pos2d is None:
                continue
            y1, x1 = from_pos2d[0], from_pos2d[1]
            y2, x2 = to_pos2d[0], to_pos2d[1]
            # Only stop drawing lines if both are outside
            if not point_in_canvas((y1, x1)) and not point_in_canvas((y2, x2)):
                continue
            for x, y in get_line(x1, y1, x2, y2):
                if point_in_canvas((y, x)):
                    array[int(y), int(x)] = (255, 0, 0)



def point_in_canvas(pos):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (pos[1] < WINDOW_WIDTH):
        return True
    return False

def get_line(x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #print("Calculating line from {},{} to {},{}".format(x1,y1,x2,y2))
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

def draw_rect(array, pos, size, color=(255, 0, 255)):
    """Draws a rect"""
    point_0 = (pos[0]-size/2, pos[1]-size/2)
    point_1 = (pos[0]+size/2, pos[1]+size/2)
    if point_in_canvas(point_0) and point_in_canvas(point_1):
        for i in range(size):
            for j in range(size):
                array[int(point_0[0]+i), int(point_0[1]+j)] = color

