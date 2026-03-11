"""
prmstealth_3d.py

3D PRM + A* for stealth planning with rotating cameras.
State space is (x, y, t).

This version assumes:
- robot moves at constant speed
- cameras rotate at constant angular speed
- nodes are sampled in (x, y, t)
- edges are valid only if the robot can move between them physically
  and is never seen during traversal
"""

import bisect
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation

from math import inf, sqrt, pi
from shapely.geometry import Point, LineString, MultiPolygon
from shapely.prepared import prep

from stealth import generate_maze, Camera


############################################################
# PARAMETERS
############################################################

N = 1500                  # number of sampled 3D nodes
K = 12                    # max outgoing neighbors
rows = 12
cols = 10

ROBOT_SPEED = 1.0         # units / second
CAMERA_OMEGA = pi / 6     # rad / second
TIME_LAYERS = 40          # sample time from these discrete layers
TIME_EPS = 0.35           # allowed mismatch between dt and d/v
WAIT_EPS = 0.20           # if spatial distance <= WAIT_EPS, allow wait edge
EDGE_CHECK_STEPS = 25     # interpolation samples per edge
CAMERA_RESOLUTION = 120   # visible polygon resolution

random.seed(9)
np.random.seed(9)

# Period of one full camera rotation
T_PERIOD = 2 * pi / abs(CAMERA_OMEGA)
TIME_SAMPLES = np.linspace(0.0, T_PERIOD, TIME_LAYERS, endpoint=False)


############################################################
# WORLD SETUP
############################################################

walls = generate_maze(rows, cols)
obstacles = prep(MultiPolygon(walls))

cameras = [
    Camera(5, 1, direction=pi, fov_angle=pi/2, max_range=5, omega=CAMERA_OMEGA),
    Camera(3, 8, direction=pi, fov_angle=pi/2, max_range=5, omega=CAMERA_OMEGA),
    Camera(9, 6, direction=pi, fov_angle=pi/2, max_range=5, omega=CAMERA_OMEGA),
]

# choose initial orientation at t = 0
for cam in cameras:
    cam.auto_orient(walls)

(xstart, ystart) = (0.5, 0.5)
(xgoal,  ygoal)  = (cols - 0.5, rows - 0.5)


############################################################
# HELPERS
############################################################

def wrap_time(t):
    """Wrap time into one camera rotation period."""
    return t % T_PERIOD


def nearest_time_sample(t):
    """
    Snap time to the nearest discrete time layer.
    """
    idx = int(np.argmin(np.abs(TIME_SAMPLES - wrap_time(t))))
    return TIME_SAMPLES[idx]


def point_seen_at_time(x, y, t):
    """
    True if point (x,y) lies inside any camera visible polygon
    at the nearest discrete sampled time.
    """
    t_snap = nearest_time_sample(t)
    p = Point(x, y)

    for poly in camera_polys_by_time[t_snap]:
        if poly.contains(p):
            return True
    return False

def get_camera_polygons_at_time(t):
    """
    Return the visible polygons of all cameras at time t.
    """
    polys = []
    for cam in cameras:
        poly = cam.visible_polygon(walls, resolution=CAMERA_RESOLUTION, t=t)
        polys.append(poly)
    return polys


def interpolate_path(path, frames_per_edge=20):
    """
    Convert a discrete node path into dense animation samples.

    Returns a list of tuples:
        (x, y, t)
    """
    samples = []

    if not path or len(path) == 0:
        return samples

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]

        dt = a.dt_forward(b)

        for k in range(frames_per_edge):
            s = k / frames_per_edge
            x = a.x + s * (b.x - a.x)
            y = a.y + s * (b.y - a.y)
            t = wrap_time(a.t + s * dt)
            samples.append((x, y, t))

    # include final point
    samples.append((path[-1].x, path[-1].y, path[-1].t))
    return samples

############################################################
# PRECOMPUTED CAMERA POLYGONS
############################################################

# map each allowed sampled time to a list of prepared camera polygons
camera_polys_by_time = {}

for t in TIME_SAMPLES:
    polys = []
    for cam in cameras:
        poly = cam.visible_polygon(walls, resolution=CAMERA_RESOLUTION, t=t)
        polys.append(prep(poly))
    camera_polys_by_time[t] = polys


############################################################
# VISUALIZATION
############################################################

class Visualization:
    def __init__(self, show_camera_time=0.0):
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

        # draw camera FOV snapshot at one time
        for cam in cameras:
            poly = cam.visible_polygon(walls, resolution=CAMERA_RESOLUTION, t=show_camera_time)
            if not poly.is_empty:
                x, y = poly.exterior.xy
                plt.fill(x, y, color='red', alpha=0.25)

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
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = wrap_time(t)
        self.neighbors = set()

        # A* fields
        self.done = False
        self.seen = False
        self.parent = None
        self.creach = 0.0
        self.ctogoest = inf

    ####################################################
    # COST / DISTANCE
    ####################################################

    def spatialDistance(self, other):
        return sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    def dt_forward(self, other):
        """
        Forward time difference on the periodic interval [0, T_PERIOD).
        """
        return (other.t - self.t) % T_PERIOD

    def required_travel_time(self, other):
        return self.spatialDistance(other) / ROBOT_SPEED

    def costToConnect(self, other):
        # optimize elapsed time
        return self.dt_forward(other)

    def costToGoEst(self, goal_xy):
        gx, gy = goal_xy
        return sqrt((gx - self.x)**2 + (gy - self.y)**2) / ROBOT_SPEED

    def __lt__(self, other):
        return (self.creach + self.ctogoest) < (other.creach + other.ctogoest)

    ####################################################
    # VALIDITY
    ####################################################

    def inFreespace(self):
        p = Point(self.x, self.y)

        # not in wall
        if not obstacles.disjoint(p):
            return False

        # not seen at this time
        if point_seen_at_time(self.x, self.y, self.t):
            return False

        return True

    def is_wait_edge(self, other):
        d = self.spatialDistance(other)
        dt = self.dt_forward(other)
        return (d <= WAIT_EPS) and (dt > 1e-9)

    def is_motion_edge(self, other):
        d = self.spatialDistance(other)
        dt = self.dt_forward(other)

        if dt <= 1e-9:
            return False

        required = d / ROBOT_SPEED
        return abs(dt - required) <= TIME_EPS

    def is_dynamically_feasible(self, other):
        return self.is_wait_edge(other) or self.is_motion_edge(other)

    def edgeIsSafe(self, other):
        """
        Check full continuous interpolation in (x,y,t).
        """
        if not self.is_dynamically_feasible(other):
            return False

        # For non-wait edges, spatial line must not hit walls
        if not self.is_wait_edge(other):
            line = LineString([(self.x, self.y), (other.x, other.y)])
            if not obstacles.disjoint(line):
                return False

        dt = self.dt_forward(other)

        for i in range(EDGE_CHECK_STEPS + 1):
            s = i / EDGE_CHECK_STEPS
            x = self.x + s * (other.x - self.x)
            y = self.y + s * (other.y - self.y)
            t = wrap_time(self.t + s * dt)

            p = Point(x, y)
            if not obstacles.disjoint(p):
                return False

            if point_seen_at_time(x, y, t):
                return False

        return True

    def __repr__(self):
        return f"<Node {self.x:.2f},{self.y:.2f},t={self.t:.2f}>"


############################################################
# PRM FUNCTIONS
############################################################

def createNodes(N):
    nodes = []
    while len(nodes) < N:
        node = Node(
            random.uniform(0, cols),
            random.uniform(0, rows),
            random.choice(TIME_SAMPLES)
        )
        if node.inFreespace():
            nodes.append(node)
    return nodes


def addGoalNodes(nodes, num_goal_times=TIME_LAYERS):
    """
    Add multiple copies of the goal at different times.
    """
    goalnodes = []
    for t in np.linspace(0.0, T_PERIOD, num_goal_times, endpoint=False):
        g = Node(xgoal, ygoal, t)
        if g.inFreespace():
            nodes.append(g)
            goalnodes.append(g)
    return goalnodes


def connectNearestNeighbors(nodes, K):
    """
    Directed 3D PRM:
    - edge direction follows forward time
    - candidate edges are ranked by timing mismatch and spatial distance
    """
    for node in nodes:
        node.neighbors = set()

    for node in nodes:
        candidates = []

        for other in nodes:
            if other is node:
                continue

            dt = node.dt_forward(other)
            if dt <= 1e-9:
                continue

            d = node.spatialDistance(other)

            if d <= WAIT_EPS:
                mismatch = 0.0
            else:
                mismatch = abs(dt - d / ROBOT_SPEED)

            candidates.append((mismatch, d, dt, other))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        added = 0
        for mismatch, d, dt, other in candidates:
            if not node.is_dynamically_feasible(other):
                continue

            if node.edgeIsSafe(other):
                node.neighbors.add(other)
                added += 1

            if added >= K:
                break


############################################################
# A*
############################################################

def astar(nodes, start, goalnodes):
    for node in nodes:
        node.done = False
        node.seen = False
        node.parent = None
        node.creach = 0.0
        node.ctogoest = inf

    goalset = set(goalnodes)
    onDeck = []

    start.seen = True
    start.creach = 0.0
    start.ctogoest = min(start.costToGoEst((xgoal, ygoal)) for _ in goalnodes)
    bisect.insort(onDeck, start)

    found_goal = None

    while onDeck:
        node = onDeck.pop(0)
        node.done = True

        if node in goalset:
            found_goal = node
            break

        for neighbor in node.neighbors:
            if neighbor.done:
                continue

            creach = node.creach + node.costToConnect(neighbor)

            if not neighbor.seen:
                neighbor.seen = True
                neighbor.parent = node
                neighbor.creach = creach
                neighbor.ctogoest = neighbor.costToGoEst((xgoal, ygoal))
                bisect.insort(onDeck, neighbor)
                continue

            if neighbor.creach <= creach:
                continue

            neighbor.parent = node
            neighbor.creach = creach

            if neighbor in onDeck:
                onDeck.remove(neighbor)
            bisect.insort(onDeck, neighbor)

    if found_goal is None:
        return None

    path = [found_goal]
    while path[0].parent:
        path.insert(0, path[0].parent)

    return path

############################################################
# ANIMATION
############################################################

def animate_path(path, frames_per_edge=20, interval=80, save=False, filename="stealth_animation.gif"):
    """
    Animate robot motion along the final path while cameras rotate.

    Parameters
    ----------
    path : list of Node
        The A* path.
    frames_per_edge : int
        Number of interpolated animation frames between roadmap nodes.
    interval : int
        Delay between frames in ms.
    save : bool
        If True, saves animation to file.
    filename : str
        Output file name if save=True.
    """
    if not path:
        print("No path to animate.")
        return

    samples = interpolate_path(path, frames_per_edge=frames_per_edge)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Temporal Stealth Animation")

    # draw walls once
    for wall in walls:
        xw, yw = wall.exterior.xy
        ax.fill(xw, yw, color='black')

    # draw start/goal
    ax.plot(xstart, ystart, 'go', markersize=10, label="Start")
    ax.plot(xgoal, ygoal, 'bo', markersize=10, label="Goal")

    # draw roadmap path statically in light blue
    for i in range(len(path) - 1):
        ax.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y],
                color='deepskyblue', linewidth=2, alpha=0.5)

    # robot marker
    robot_plot, = ax.plot([], [], 'mo', markersize=8, label="Robot")

    # trail
    trail_plot, = ax.plot([], [], 'm-', linewidth=2, alpha=0.8)

    # camera centers
    for cam in cameras:
        ax.plot(cam.x, cam.y, 'ro', markersize=6)

    # dynamic camera fills and heading lines
    cam_patches = []
    cam_headings = []
    for cam in cameras:
        patch = ax.fill([], [], color='red', alpha=0.25)[0]
        cam_patches.append(patch)

        heading_line, = ax.plot([], [], 'r-', linewidth=2)
        cam_headings.append(heading_line)

    time_text = ax.text(0.02, 1.02, "", transform=ax.transAxes, fontsize=11)
    detect_text = ax.text(0.55, 1.02, "", transform=ax.transAxes, fontsize=11, color='darkred')

    xs_trail = []
    ys_trail = []

    def init():
        robot_plot.set_data([], [])
        trail_plot.set_data([], [])
        time_text.set_text("")
        detect_text.set_text("")

        for patch in cam_patches:
            patch.set_xy(np.empty((0, 2)))

        for line in cam_headings:
            line.set_data([], [])

        return [robot_plot, trail_plot, time_text, detect_text] + cam_patches + cam_headings

    def update(frame):
        x, y, t = samples[frame]

        xs_trail.append(x)
        ys_trail.append(y)

        robot_plot.set_data([x], [y])
        trail_plot.set_data(xs_trail, ys_trail)

        # update camera FOV polygons
        polys = get_camera_polygons_at_time(t)
        robot_seen = False

        for idx, (cam, poly, patch, heading_line) in enumerate(zip(cameras, polys, cam_patches, cam_headings)):
            if not poly.is_empty and hasattr(poly, "exterior"):
                coords = np.column_stack(poly.exterior.xy)
                patch.set_xy(coords)
            else:
                patch.set_xy(np.empty((0, 2)))

            theta = cam.direction_at(t)
            hx = cam.x + 0.5 * np.cos(theta)
            hy = cam.y + 0.5 * np.sin(theta)
            heading_line.set_data([cam.x, hx], [cam.y, hy])

            if poly.contains(Point(x, y)):
                robot_seen = True

        time_text.set_text(f"time = {t:.2f} s")
        detect_text.set_text("DETECTED" if robot_seen else "hidden")

        return [robot_plot, trail_plot, time_text, detect_text] + cam_patches + cam_headings

    anim = FuncAnimation(
        fig,
        update,
        frames=len(samples),
        init_func=init,
        interval=interval,
        blit=False,
        repeat=False
    )

    ax.legend()

    if save:
        anim.save(filename, dpi=120)
        print(f"Animation saved to {filename}")

    plt.show()

############################################################
# MAIN
############################################################

def main():
    print("Running 3D temporal PRM")
    print("N =", N, "K =", K)
    print("Robot speed =", ROBOT_SPEED)
    print("Camera omega =", CAMERA_OMEGA)
    print("Time period =", T_PERIOD)

    visual = Visualization(show_camera_time=0.0)

    startnode = Node(xstart, ystart, 0.0)
    if not startnode.inFreespace():
        print("Start node is invalid at t = 0")
        return

    visual.drawNode(startnode, color='orange', marker='o', markersize=8)
    visual.show("World ready")

    print("Sampling 3D nodes...")
    nodes = createNodes(N)

    for node in nodes:
        visual.drawNode(node, color='k', marker='x', markersize=3)

    nodes.append(startnode)

    goalnodes = addGoalNodes(nodes, num_goal_times=TIME_LAYERS)

    if not goalnodes:
        print("No valid goal-time nodes found")
        return

    for g in goalnodes:
        visual.drawNode(g, color='purple', marker='o', markersize=4)

    visual.show("Nodes sampled")

    print("Connecting neighbors...")
    connectNearestNeighbors(nodes, K)

    edge_count = 0
    for i, node in enumerate(nodes):
        for neighbor in node.neighbors:
            edge_count += 1
            visual.drawEdge(node, neighbor, color='green', linewidth=0.2)

    print("Directed edges:", edge_count)
    visual.show("Graph built")

    print("Running A*...")
    path = astar(nodes, startnode, goalnodes)

    if not path:
        print("NO PATH FOUND")
        return

    print("PATH FOUND")
    print("Elapsed time cost:", path[-1].creach)
    print("Arrival phase:", path[-1].t)

    for p in path:
        print(p)

    visual.drawPath(path, color='blue', linewidth=3)
    visual.show("Temporal stealth path found")

    animate_path(path, frames_per_edge=25, interval=80)


if __name__ == "__main__":
    main()
