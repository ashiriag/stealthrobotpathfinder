"""
prmstealth.py

PRM + A* with camera exposure penalty.
Uses maze + cameras from stealth.py
"""

import bisect
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math import inf, sqrt, pi
from shapely.geometry import Point, LineString, MultiPolygon
from shapely.prepared import prep

from stealth import *   # generate_maze, Camera


############################################################
# PARAMETERS
############################################################

N = 400
K = 8
LAMBDA = 8      # stealth penalty strength (tune this!)

rows = 12
cols = 10


############################################################
# WORLD SETUP
############################################################

walls = generate_maze(rows, cols)
obstacles = prep(MultiPolygon(walls))

cameras = [
    Camera(5, 1, direction=pi, fov_angle=pi/2, max_range=5),
    Camera(3, 8, direction=pi, fov_angle=pi/2, max_range=5),
    Camera(9, 6, direction=pi, fov_angle=pi/2, max_range=5),
]

camera_polys = [cam.visible_polygon(walls, resolution=180) for cam in cameras]

(xstart, ystart) = (0.5, 0.5)
(xgoal,  ygoal)  = (cols - 0.5, rows - 0.5)


############################################################
# VISUALIZATION
############################################################

class Visualization:
    def __init__(self):
        plt.clf()
        plt.axes()
        plt.grid(True)
        plt.gca().set_xlim(0, cols)
        plt.gca().set_ylim(0, rows)
        plt.gca().set_aspect('equal')

        # draw walls
        for wall in walls:
            x, y = wall.exterior.xy
            plt.fill(x, y, color='black')

        # draw camera FOV
        for cam, poly in zip(cameras, camera_polys):
            x, y = poly.exterior.xy
            plt.fill(x, y, color='red', alpha=0.3)
            plt.plot(cam.x, cam.y, 'ro')

        plt.pause(0.001)

    def show(self, text=''):
        plt.pause(0.001)
        if text:
            input(text + ' (hit return)')

    def drawNode(self, node, **kwargs):
        plt.plot(node.x, node.y, **kwargs)

    def drawEdge(self, a, b, **kwargs):
        plt.plot([a.x, b.x], [a.y, b.y], **kwargs)

    def drawPath(self, path, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], **kwargs)


############################################################
# NODE
############################################################

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbors = set()

        # A* fields
        self.done = False
        self.seen = False
        self.parent = None
        self.creach = 0
        self.ctogoest = inf

    ####################################################
    # COST FUNCTIONS
    ####################################################

    def distance(self, other):
        return sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    def exposurePenalty(self, other):
        line = LineString([(self.x, self.y), (other.x, other.y)])
        penalty = 0
        for poly in camera_polys:
            if line.intersects(poly):
                inter = line.intersection(poly)
                penalty += inter.length
        return penalty

    def costToConnect(self, other):
        base = self.distance(other)
        penalty = self.exposurePenalty(other)
        return base + LAMBDA * penalty

    def costToGoEst(self, goal):
        # Heuristic = pure distance (admissible)
        return self.distance(goal)

    def __lt__(self, other):
        return (self.creach + self.ctogoest) < (other.creach + other.ctogoest)

    ####################################################
    # PRM FUNCTIONS
    ####################################################

    def inFreespace(self):
        return obstacles.disjoint(Point(self.x, self.y))

    def connectsTo(self, other):
        line = LineString([(self.x, self.y), (other.x, other.y)])
        return obstacles.disjoint(line)

    def __repr__(self):
        return f"<Node {self.x:.2f},{self.y:.2f}>"


############################################################
# PRM FUNCTIONS
############################################################

def createNodes(N):
    nodes = []
    while len(nodes) < N:
        node = Node(random.uniform(0, cols),
                    random.uniform(0, rows))
        if node.inFreespace():
            nodes.append(node)
    return nodes


def connectNearestNeighbors(nodes, K):
    for node in nodes:
        node.neighbors = set()

    for node in nodes:
        distances = np.array([node.distance(n) for n in nodes])
        indicies = np.argsort(distances)

        for k in indicies[1:K+1]:
            neighbor = nodes[k]
            if neighbor not in node.neighbors and node.connectsTo(neighbor):
                node.neighbors.add(neighbor)
                neighbor.neighbors.add(node)


############################################################
# A*
############################################################

def astar(nodes, start, goal):

    for node in nodes:
        node.done = False
        node.seen = False
        node.parent = None
        node.creach = 0
        node.ctogoest = inf

    onDeck = []

    start.seen = True
    start.ctogoest = start.costToGoEst(goal)
    bisect.insort(onDeck, start)

    while onDeck:

        node = onDeck.pop(0)
        node.done = True

        if node == goal:
            break

        for neighbor in node.neighbors:

            if neighbor.done:
                continue

            creach = node.creach + node.costToConnect(neighbor)

            if not neighbor.seen:
                neighbor.seen = True
                neighbor.parent = node
                neighbor.creach = creach
                neighbor.ctogoest = neighbor.costToGoEst(goal)
                bisect.insort(onDeck, neighbor)
                continue

            if neighbor.creach <= creach:
                continue

            neighbor.parent = node
            neighbor.creach = creach
            onDeck.remove(neighbor)
            bisect.insort(onDeck, neighbor)

    if not goal.parent:
        return None

    path = [goal]
    while path[0].parent:
        path.insert(0, path[0].parent)

    return path


############################################################
# MAIN
############################################################

def main():

    print("Running PRM stealth with N =", N, "K =", K, "Lambda =", LAMBDA)

    visual = Visualization()

    startnode = Node(xstart, ystart)
    goalnode = Node(xgoal, ygoal)

    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode, color='purple', marker='o')
    visual.show("World ready")

    print("Sampling nodes...")
    nodes = createNodes(N)

    for node in nodes:
        visual.drawNode(node, color='k', marker='x')

    nodes.append(startnode)
    nodes.append(goalnode)

    visual.show("Nodes sampled")

    print("Connecting neighbors...")
    connectNearestNeighbors(nodes, K)

    for i,node in enumerate(nodes):
        for neighbor in node.neighbors:
            if neighbor not in nodes[:i]:
                visual.drawEdge(node, neighbor, color='green', linewidth=0.3)

    visual.show("Graph built")

    print("Running A*...")
    path = astar(nodes, startnode, goalnode)

    if not path:
        print("NO PATH FOUND")
        return

    visual.drawPath(path, color='blue', linewidth=3)
    visual.show("Stealth path found")


if __name__ == "__main__":
    main()