#!/usr/bin/env python
import rospy
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import median_filter
from scipy.stats import linregress
import math
import time
from matplotlib import pyplot as plt
import operator
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped

def get_robot_pose(data):
    global robot_position

    robot_position = data

def get_path_map(data):
    global path_map

    path_map = raw_to_path(data.data, data.info.width, data.info.height, data.info.resolution)

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

    ordered_goals = shortest_path(unordered_goals)

    write_image((path_map.copy(), path_map.copy(), path_map.copy()), ordered_goals)

    return [tuple(map(operator.add, goal, (map_origin_y, map_origin_x, 0))) for goal in ordered_goals]

def write_image((r, g, b), goals):
    # Show Goals + Path
    pass

def raw_to_path(raw_data, raw_width, raw_height, resolution):
    processed_map = median_filter(np.flipud(np.reshape(np.matrix(raw_data, dtype=np.uint8), (raw_width, raw_height)))[map_origin_y:map_origin_y + map_height, map_origin_x:map_origin_x + map_width], size=median_filter_size / resolution)
    processed_map[processed_map == 0] = 1
    processed_map[processed_map == -1] = 0
    processed_map[processed_map == 100] = 0

    processed_map = distance_transform_edt(processed_map)
    processed_map[processed_map * resolution > max_wall_distance] = 0
    processed_map[processed_map * resolution < min_wall_distance] = 0
    processed_map[processed_map > 0] = 1

    return processed_map

def get_goal_from_chunk(chunk, start_x, start_y, unordered_goals):
    path = zip(chunk[0], chunk[1])
    median_position = None
    angle = 90 - math.atan(linregress(chunk[0], chunk[1])[0]) * (180 / math.pi)

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

    if math.isnan(angle) and median_position is not None:
        closest_goal = nearest_neighbor(median_position, unordered_goals)
        angle = 90 - math.atan((closest_goal[0] - median_position[0]) / (closest_goal[1] - median_position[1])) * (180 / math.pi)

    return None if median_position is None else (median_position[0], median_position[1], angle)

def distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def nearest_neighbor(A, goals):
    "Find the goal in goals that is nearest to goal A."
    return min(goals, key=lambda x: distance(x, A))

def shortest_path(goals, start=None):
    """At each step, call the last waypoint in the path A, and consider the
    nearest neighbor B, and also any waypoint D that has A as nearest neighbor."""
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

if __name__ == '__main__':
    rospy.init_node('wall_scanner_planner')

    robot_position = (0,0,0) #None
    path_map = None
    goals = []

    # Map cropping
    map_origin_x = rospy.get_param('~map_origin_x', 500)
    map_origin_y = rospy.get_param('~map_origin_y', 700)
    map_width = rospy.get_param('~map_width', 1200)
    map_height = rospy.get_param('~map_height', 500)

    # Specs provided by camera
    min_wall_distance = rospy.get_param('~min_wall_distance', 0.8)  #0.8
    max_wall_distance = rospy.get_param('~max_wall_distance', 0.9)  #3.5

    # Map Smoothing
    median_filter_size = rospy.get_param('~median_filter_size', 0.5)

    # Number of waypoints
    chunk_width = rospy.get_param('~chunk_width', 30)
    chunk_height = rospy.get_param('~chunk_height', 30)
    min_waypoint_distance = math.sqrt(chunk_width ** 2 + chunk_height ** 2)
    #processable_chunk_threshold = rospy.get_param('~processable_chunk_threshold', 0.01)

    rospy.Subscriber("/map", OccupancyGrid, get_path_map)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, get_robot_pose)

    while not rospy.is_shutdown():
        if len(goals) == 0:
            if robot_position is not None and path_map is not None:
                goals = get_goals()

                print "\n\n\n"
                print goals
