#!/usr/bin/env python
import rospy
import actionlib
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseGoal

import math
from random import randint

import numpy as np
import operator
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import median_filter

import Image, ImageDraw, ImageOps

def get_path_map(data):
    global path_map, wall_map

    path_map, wall_map = raw_data_to_maps(data.data, data.info.width, data.info.height, data.info.resolution)

def get_goals():
    unordered_goals = set()

    for x1 in xrange(0, map_width, chunk_width):
        x2 = x1 + chunk_width if x1 + chunk_width <= map_width else map_width

        for y1 in xrange(0, map_height, chunk_height):
            y2 = y1 + chunk_height if y1 + chunk_height <= map_height else map_height
            chunk = path_map[y1:y2, x1:x2]

            if np.count_nonzero(chunk) > 0: #math.floor((chunk_width * chunk_height) * processable_chunk_threshold):
                goal = get_goal_from_chunk(np.where(chunk != 0), x1, y1, unordered_goals)

                if goal is not None:
                    unordered_goals.add(goal)

    #ordered_goals = shortest_path(unordered_goals)
    #ordered_goals = mst(unordered_goals, robot_position)


    #pprint(ordered_goals)

    starting_point = (None, 99999)

    for goal in unordered_goals:
        path = shortest_path(unordered_goals.copy(), goal)
        total_distance = get_path_length(path)

        print 'SP: %f' % (total_distance)

        if starting_point[1] > total_distance:
            starting_point = (goal, total_distance)

    ordered_goals = shortest_path(unordered_goals.copy(), starting_point[0])
    set_goal_angles(ordered_goals)
    write_image((path_map.copy(), path_map.copy(), path_map.copy()), ordered_goals, "path_best-sp.png", True)
    write_image((wall_map.copy(), wall_map.copy(), wall_map.copy()), ordered_goals, "walls_best-sp.png")
    #
    # ordered_goals = prim(unordered_goals.copy())
    # set_goal_angles(ordered_goals)
    # write_image((path_map.copy(), path_map.copy(), path_map.copy()), ordered_goals, "path-prim.png", True)
    #     #write_image((wall_map.copy(), wall_map.copy(), wall_map.copy()), ordered_goals, "walls-prim.png")
    # print 'Prim: %f' % (get_path_length(ordered_goals))
    #
    # ordered_goals = mst(unordered_goals.copy(), robot_position)
    # set_goal_angles(ordered_goals)
    # write_image((path_map.copy(), path_map.copy(), path_map.copy()), ordered_goals, "path-mst.png", True)
    #     #write_image((wall_map.copy(), wall_map.copy(), wall_map.copy()), ordered_goals, "walls-mst.png")
    # print 'MST: %f' % (get_path_length(ordered_goals))
    #
    # ordered_goals = shortest_path(unordered_goals.copy())
    # set_goal_angles(ordered_goals)
    # write_image((path_map.copy(), path_map.copy(), path_map.copy()), ordered_goals, "path-sp.png", True)
    #     #write_image((wall_map.copy(), wall_map.copy(), wall_map.copy()), ordered_goals, "walls-sp.png")
    # print 'SP: %f' % (get_path_length(ordered_goals))

    return [tuple(map(operator.add, goal, (map_origin_y, map_origin_x, 0))) for goal in ordered_goals]

def get_path_length(path):
    total_distance = 0
    previous_goal = None

    for goal in path:
        if previous_goal is not None:
            total_distance += distance(previous_goal, goal)

        previous_goal = goal

    return total_distance

#######################################
#
# Start: PRIM'S ALGORITHM
#

def prim(vertices):
    """Given a set of cities, build a minimum spanning tree: a dict of the form {parent: [child...]},
    where parent and children are cities, and the root of the tree is first(cities)."""
    N = len(vertices)
    start = min(vertices, key=lambda x: distance(x, robot_position))
    edges = shortest_first([(A, B) for A in vertices for B in vertices if A is not B])
    tree = {start: []} # the first city is the root of the tree.

    while len(tree) < N:
        (A, B) = first((A, B) for (A, B) in edges if (A in tree) and (B not in tree))
        tree[A].append(B)
        tree[B] = []

    return preorder_uniq(tree, start, [])

def shortest_first(edges):
    "Sort a list of edges so that shortest come first."
    edges.sort(key=lambda (A, B): distance(A, B))
    return edges

def first(collection):
    "Start iterating over collection, and return the first element."
    for x in collection: return x

def preorder_uniq(tree, node, result):
    "Traverse tree in pre-order, starting at node, omitting repeated nodes."
    # Accumulate results in the 'result' parameter, which should start with an empty list
    if node not in result:
        result.append(node)
    for child in tree[node]:
        preorder_uniq(tree, child, result)
    return result

#
# End: PRIM'S ALGORITHM
#
#######################################
#
# Start: SIMPLE MINIMUM SPANNING TREE
#

def mst(vertices, starting_position):
    edges = set()
    first = min(vertices, key=lambda x: distance(x, starting_position))
    path = [first]

    for v1 in vertices:
        for v2 in neighbors(v1, vertices):
            if (v1, v2[0], v2[1]) not in edges and (v2[0], v1, v2[1]) not in edges:
                edges.add((v1,) + v2)

    vertices.remove(first)

    while len(vertices) > 0:
        e = min(set([(v1, v2, d) for v1,v2, d in edges if (v1 in path and v2 not in path) or (v1 not in path and v2 in path)]), key=lambda x: x[2])
        v = e[0] if e[0] not in path else e[1]

        path.append(v)
        vertices.remove(v)

    return path

def neighbors(x, vertices):
    n = set()

    for v in vertices:
        d = distance(x, v)

        if x != v: #and d < 60:#1.5 * min_waypoint_distance:
            n.add((v, d))

    return n

#
# End: SIMPLE MINIMUM SPANNING TREE
#
#######################################
#
# Start: DOUBLE GREEDY SHORTEST PATH
#

def shortest_path(goals, start=None):
    NN = {C: nearest_neighbor(C, goals - {C}) for C in goals}

    if start is None:
        start = nearest_neighbor(robot_position, goals)

    path = [start]
    unvisited = goals - {start}

    while unvisited:
        A = path[-1]
        B = NN[A] if NN[A] in unvisited else nearest_neighbor(A, unvisited)
        Ds = [D for D in unvisited if NN[D] is A and D is not B]
        C = (min(Ds, key=lambda D: distance(D, A))) if Ds else B

        path.append(C)
        unvisited.remove(C)

    return path

def nearest_neighbor(A, goals):

    return min(goals, key=lambda x: distance(x, A))

#
# End: DOUBLE GREEDY SHORTEST PATH
#
#######################################

def write_image((r, g, b), goals, filename, invert=False):
    a = np.ones(r.shape, dtype=np.uint8)
    a[a == 1] = 255

    rgba_path = Image.fromarray(np.uint8(np.dstack((r, g, b, a))), 'RGBA')
    r, g, b, a = rgba_path.split()
    rgb_path = Image.merge('RGB', (r,g,b))

    if invert:
        rgb_path = ImageOps.invert(rgb_path)

    r, g, b = rgb_path.split()

    path = Image.merge('RGBA', (r, g, b, a))
    draw = ImageDraw.Draw(path)
    arrow = Image.new('RGBA', (20, 20))
    draw_arrow = ImageDraw.Draw(arrow)
    previous_goal = None

    draw_arrow.rectangle((8,20,12,12), fill=0xAAFF0000)
    draw_arrow.polygon((4,12,10,0,16,12), fill=0xAAFF0000)

    for i, goal in enumerate(goals):
        if previous_goal is not None:
            draw.line((previous_goal[1], previous_goal[0], goal[1], goal[0]), fill=0xFF000080)

        rot = arrow.rotate( goal[2], expand=1 )
        path.paste( rot, (goal[1] - 10, goal[0] - 10), rot )
        draw.ellipse((goal[1] - 2, goal[0] - 2, goal[1] + 2, goal[0] + 2), fill=0xFF0000FF if i > 0 else 0xFF00FF00)
        previous_goal = goal

    path.save(filename)

def raw_data_to_maps(raw_data, raw_width, raw_height, resolution):
    cropped_map = np.flipud(np.reshape(np.matrix(raw_data, dtype=np.uint8), (raw_width, raw_height)))[map_origin_y:map_origin_y + map_height, map_origin_x:map_origin_x + map_width]
    walls = cropped_map.copy()

    cropped_map[cropped_map > 0] = 1
    walls = median_filter(walls, size=(median_filter_size / resolution))
    walls[walls == 0] = 255
    walls[walls == 100] = 0

    walls[walls == 1] = 0
    walls[walls == 157] = 0
    path = distance_transform_edt(1 - median_filter(cropped_map, size=median_filter_size / resolution))
    path[path * resolution > max_wall_distance] = 0
    path[path * resolution < min_wall_distance] = 0
    path[path > 0] = 0x0000FF

    return path, walls

def get_goal_from_chunk(chunk, start_x, start_y, unordered_goals):
    path = zip(chunk[0], chunk[1])
    median_position = None

    for pp in path:
        total_distance = 0
        valid = True
        p = (pp[0] + start_y, pp[1] + start_x)

        for g in unordered_goals:
            d = distance(p, g)

            if d > min_waypoint_distance:
                total_distance += d
            else:
                valid = False
                break

        if valid and (median_position is None or median_position[2] < total_distance):
            median_position = (pp[0] + start_y, pp[1] + start_x, total_distance)

    return None if median_position is None else (median_position[0], median_position[1], 0)

def distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def set_goal_angles(goals):
    for i, goal in enumerate(goals):
        goals[i - 1] = (goals[i - 1][0], goals[i - 1][1], (math.atan2(goal[1] - goals[i - 1][1], goal[0] - goals[i - 1][0]) + math.pi) * 360.0 / (2.0 * math.pi))

if __name__ == '__main__':
    rospy.init_node('wall_explorer')

    robot_position = (randint(0,500), randint(0,1200), 0) #None
    wall_map = None
    path_map = None
    goals = []

    # Map cropping
    map_origin_x = rospy.get_param('~map_origin_x', 500)
    map_origin_y = rospy.get_param('~map_origin_y', 700)
    map_width = rospy.get_param('~map_width', 1200)
    map_height = rospy.get_param('~map_height', 500)

    # Specs provided by camera
    min_wall_distance = rospy.get_param('~min_wall_distance', 0.8)  #0.8
    max_wall_distance = rospy.get_param('~max_wall_distance', 0.85)  #3.5

    # Map Smoothing
    median_filter_size = rospy.get_param('~median_filter_size', 0.5)

    # Number of waypoints
    chunk_width = rospy.get_param('~chunk_width', 25)
    chunk_height = rospy.get_param('~chunk_height', 25)
    min_waypoint_distance = math.sqrt(chunk_width ** 2 + chunk_height ** 2)
    #processable_chunk_threshold = rospy.get_param('~processable_chunk_threshold', 0.01)

    rospy.Subscriber("/map", OccupancyGrid, get_path_map)
    goal_publisher = rospy.Publisher('/wall_explorer/goals', MoveBaseGoal, queue_size=100)

    while not rospy.is_shutdown():
        if len(goals) == 0:
            if robot_position is not None and path_map is not None:
                goals = get_goals()
