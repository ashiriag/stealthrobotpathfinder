# rrtstealth.py
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import Point, LineString, MultiPolygon
from shapely.prepared import prep
from math import sqrt, pi

from stealth import *  # Your maze + Camera code


##############################
# Node Definition for RRT
##############################
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def inFreespace(self, obstacles, rows, cols):
        # Check bounds
        if not (0 <= self.x <= cols and 0 <= self.y <= rows):
            return False
        return obstacles.disjoint(Point(self.x, self.y))

    def connectsTo(self, other, obstacles):
        return obstacles.disjoint(LineString([(self.x, self.y), (other.x, other.y)]))


##############################
# RRT Algorithm
##############################
def rrt(startnode, goalnode, obstacles, rows, cols, DSTEP=0.8, NMAX=1500, SMAX=50000, visual=None):
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
        # 5% goal bias
        if random.random() < 0.05:
            target = goalnode
        else:
            target = Node(random.uniform(0, cols), random.uniform(0, rows))

        # Find closest node
        closest = min(tree, key=lambda n: n.distance(target))
        dx, dy = target.x - closest.x, target.y - closest.y
        d = sqrt(dx**2 + dy**2)
        step = min(DSTEP, d)
        newnode = Node(closest.x + step*dx/d, closest.y + step*dy/d)

        if not newnode.inFreespace(obstacles, rows, cols):
            continue
        if not closest.connectsTo(newnode, obstacles):
            continue

        addtotree(closest, newnode)

        if newnode.distance(goalnode) <= DSTEP and newnode.connectsTo(goalnode, obstacles):
            addtotree(newnode, goalnode)
            break

        if steps >= SMAX or len(tree) >= NMAX:
            print(f"Aborted: steps={steps}, nodes={len(tree)}")
            return None, tree

    # Build path
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    print(f"Finished RRT: steps={steps}, nodes={len(tree)}")
    return path, tree


##############################
# Visualization Helper
##############################
class VisualWrapper:
    def __init__(self, walls, cameras, start, goal, rows, cols):
        self.walls = walls
        self.cameras = cameras
        self.start = start
        self.goal = goal
        self.rows = rows
        self.cols = cols
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, cols)
        self.ax.set_ylim(0, rows)
        self.ax.grid(True)

        # Draw walls
        for wall in walls:
            x, y = wall.exterior.xy
            self.ax.fill(x, y, color='black')

        # Draw cameras
        for cam in cameras:
            poly = cam.visible_polygon(walls, resolution=180)  # Shapely Polygon
            x_poly, y_poly = poly.exterior.xy
            self.ax.fill(x_poly, y_poly, color='red', alpha=0.3)
            self.ax.plot(cam.x, cam.y, 'ro')

        # Draw start & goal
        self.ax.plot(start[0], start[1], 'go', markersize=10)
        self.ax.plot(goal[0], goal[1], 'bo', markersize=10)

    def drawEdge(self, n1, n2, **kwargs):
        self.ax.plot([n1.x, n2.x], [n1.y, n2.y], **kwargs)
        plt.pause(0.001)

    def drawPath(self, path, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], **kwargs)


##############################
# Main Function
##############################
def main():
    random.seed(9)
    np.random.seed(9)

    rows = 12
    cols = 10

    # Generate maze
    walls = generate_maze(rows, cols)

    # Place cameras
    cameras = [
        Camera(5, 1, direction=pi, fov_angle=pi/2, max_range=5),
        Camera(3, 8, direction=pi, fov_angle=pi/2, max_range=5),
        Camera(9, 6, direction=pi, fov_angle=pi/2, max_range=5),
    ]

    start = (0.5, 0.5)
    goal = (cols - 0.5, rows - 0.5)

    # Convert maze walls to obstacles for RRT
    obstacles = prep(MultiPolygon(walls))

    # Visualization
    visual = VisualWrapper(walls, cameras, start, goal, rows, cols)

    # Run RRT
    startnode = Node(*start)
    goalnode = Node(*goal)
    path, tree = rrt(startnode, goalnode, obstacles, rows, cols, DSTEP=0.8, visual=visual)

    if path:
        # Draw RRT path
        visual.drawPath(path, color='b', linewidth=2)
        plt.show()
    else:
        print("Failed to find path.")


if __name__ == "__main__":
    main()