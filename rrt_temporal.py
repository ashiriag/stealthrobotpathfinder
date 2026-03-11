import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import Point, LineString, MultiPolygon
from shapely.prepared import prep
from math import sqrt, pi, cos, sin

from stealth import *  # maze generation

ROBOT_SPEED = 1.0
TIME_STEP_CHECKS = 10

##############################
# Rotating Camera
##############################
class RotatingCamera(Camera):
    def __init__(self, x, y, direction=0, fov_angle=pi/2, max_range=5, angular_speed=0.5):
        super().__init__(x, y, direction, fov_angle, max_range)
        self.angular_speed = angular_speed
        self.direction0 = direction

    def direction_at_time(self, t):
        return self.direction0 + self.angular_speed * t

    def visible_polygon_at_time(self, walls, t, resolution=60):
        # compute temporary direction without permanently modifying camera
        original_dir = self.direction
        self.direction = self.direction_at_time(t)
        poly = self.visible_polygon(walls, resolution)
        self.direction = original_dir
        return poly


##############################
# Node Definition for RRT
##############################
class Node:
    def __init__(self, x, y, t=0.0):
        self.x = x
        self.y = y
        self.t = t
        self.parent = None

    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


##############################
# Camera visibility test
##############################
def visible_by_camera(point, t, cameras, walls):
    for cam in cameras:
        poly = cam.visible_polygon_at_time(walls, t)
        if poly.contains(point):
            return True
    return False


##############################
# Node free space test
##############################
def node_in_free_space(node, obstacles, cameras, walls, rows, cols):
    if not (0 <= node.x <= cols and 0 <= node.y <= rows):
        return False

    pt = Point(node.x, node.y)

    if not obstacles.disjoint(pt):
        return False

    if visible_by_camera(pt, node.t, cameras, walls):
        return False

    return True


##############################
# Edge collision check
##############################
def edge_valid(n1, n2, obstacles, cameras, walls):

    line = LineString([(n1.x, n1.y), (n2.x, n2.y)])

    if not obstacles.disjoint(line):
        return False

    for a in np.linspace(0, 1, TIME_STEP_CHECKS):
        x = n1.x + a * (n2.x - n1.x)
        y = n1.y + a * (n2.y - n1.y)
        t = n1.t + a * (n2.t - n1.t)

        if visible_by_camera(Point(x, y), t, cameras, walls):
            return False

    return True


##############################
# RRT Algorithm (Space-Time)
##############################
def rrt(startnode, goalnode, obstacles, cameras, walls, rows, cols,
        DSTEP=0.8, NMAX=1500, SMAX=50000, visual=None):

    startnode.parent = None
    tree = [startnode]
    steps = 0

    def addtotree(parent, child):
        child.parent = parent
        tree.append(child)
        if visual:
            visual.drawEdge(parent, child, color='g', linewidth=1)

    while True:

        steps += 1

        if random.random() < 0.15:
            target = goalnode
        else:
            target = Node(random.uniform(0, cols), random.uniform(0, rows))

        closest = min(tree, key=lambda n: sqrt((n.x-target.x)**2 + (n.y-target.y)**2))

        dx = target.x - closest.x
        dy = target.y - closest.y

        d = sqrt(dx**2 + dy**2)

        if d == 0:
            continue

        step = min(DSTEP, d)

        newx = closest.x + step * dx / d
        newy = closest.y + step * dy / d

        travel_time = step / ROBOT_SPEED
        newt = closest.t + travel_time

        newnode = Node(newx, newy, newt)

        if not node_in_free_space(newnode, obstacles, cameras, walls, rows, cols):
            continue

        if not edge_valid(closest, newnode, obstacles, cameras, walls):
            continue

        addtotree(closest, newnode)

        if sqrt((newnode.x-goalnode.x)**2 + (newnode.y-goalnode.y)**2) <= DSTEP:

            goal_time = newnode.t + newnode.distance(goalnode) / ROBOT_SPEED
            goalnode.t = goal_time

            if edge_valid(newnode, goalnode, obstacles, cameras, walls):
                addtotree(newnode, goalnode)
                break

        if steps >= SMAX or len(tree) >= NMAX:
            print("Aborted")
            return None, tree

    path = [goalnode]

    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    print("Finished RRT")

    return path, tree


##############################
# Visualization
##############################
class VisualWrapper:

    def __init__(self, walls, cameras, start, goal, rows, cols):

        self.walls = walls
        self.cameras = cameras
        self.rows = rows
        self.cols = cols

        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, cols)
        self.ax.set_ylim(0, rows)
        self.ax.grid(True)

        for wall in walls:
            x, y = wall.exterior.xy
            self.ax.fill(x, y, color='black')

        # camera graphics
        self.cam_patches = []
        self.cam_points = []

        for cam in cameras:
            poly = cam.visible_polygon_at_time(walls, 0)
            x_poly, y_poly = poly.exterior.xy
            patch = self.ax.fill(x_poly, y_poly, color='red', alpha=0.25)[0]
            pt = self.ax.plot(cam.x, cam.y, 'ro')[0]
            self.cam_patches.append(patch)
            self.cam_points.append(pt)

        self.ax.plot(start[0], start[1], 'go', markersize=10)
        self.ax.plot(goal[0], goal[1], 'bo', markersize=10)

    def update_cameras(self, t):
        # redraw FOVs according to time
        for i, cam in enumerate(self.cameras):
            poly = cam.visible_polygon_at_time(self.walls, t)
            x_poly, y_poly = poly.exterior.xy
            self.cam_patches[i].set_xy(np.column_stack((x_poly, y_poly)))

    def drawEdge(self, n1, n2, **kwargs):
        self.ax.plot([n1.x, n2.x], [n1.y, n2.y], **kwargs)
        self.update_cameras(n2.t)
        plt.pause(0.001)

    def drawPath(self, path, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], **kwargs)


##############################
# Main
##############################

def main():

    random.seed(9)
    np.random.seed(9)

    rows = 12
    cols = 10

    walls = generate_maze(rows, cols)

    cameras = [
        RotatingCamera(5,1,direction=pi,fov_angle=pi/2,max_range=5,angular_speed=0.8),
        RotatingCamera(3,8,direction=pi/2,fov_angle=pi/2,max_range=5,angular_speed=-0.6),
        RotatingCamera(9,6,direction=0,fov_angle=pi/2,max_range=5,angular_speed=0.7)
    ]

    start = (0.5,0.5)
    goal = (cols-0.5, rows-0.5)

    obstacles = prep(MultiPolygon(walls))

    visual = VisualWrapper(walls, cameras, start, goal, rows, cols)

    startnode = Node(*start, t=0)
    goalnode = Node(*goal)

    path, tree = rrt(startnode, goalnode, obstacles, cameras, walls, rows, cols, visual=visual)

    if path:
        visual.drawPath(path, color='b', linewidth=2)
        plt.show()

    else:
        print("No path found")


if __name__ == "__main__":
    main()
