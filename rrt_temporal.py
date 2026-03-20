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

import argparse
import bisect
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib.animation import FuncAnimation

from math import inf, sqrt, pi
from shapely.geometry import Point, LineString, MultiPolygon
from shapely.prepared import prep

from stealth import generate_maze, Camera


############################################################
# PARAMETERS
############################################################
DSTEP = 5.0

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 50000
NMAX = 1500


######################################################################
#
#   World Definitions (No Fixes Needed)
#
#   List of obstacles/objects as well as the start/goal.
#
(xmin, xmax) = (0, 10)
(ymin, ymax) = (0, 12)


rows = 12
cols = 10

ROBOT_SPEED = 1.0         # units / second
CAMERA_OMEGA = pi / 6     # rad / second
TIME_LAYERS = 40          # sample time from these discrete layers
TIME_EPS = 0.35           # allowed mismatch between dt and d/v
WAIT_EPS = 0.20           # if spatial distance <= WAIT_EPS, allow wait edge
EDGE_CHECK_STEPS = 25     # interpolation samples per edge
CAMERA_RESOLUTION = 120   # visible polygon resolution
GOAL_BIAS = 0.08
MERGE_POS_EPS = 0.25

random.seed(9)
np.random.seed(9)

# Period of one full camera rotation
T_PERIOD = 2 * pi / abs(CAMERA_OMEGA)
TIME_SAMPLES = np.linspace(0.0, T_PERIOD, TIME_LAYERS, endpoint=False)
WAIT_STEP = T_PERIOD / TIME_LAYERS


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

camera_polys_by_time = {}


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


def already_in_tree(tree, node):
    for other in tree:
        if other.spatialDistance(node) <= MERGE_POS_EPS:
            return True
    return False


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


def rebuild_time_layers(time_layers=None):
    global TIME_LAYERS, TIME_SAMPLES, WAIT_STEP, camera_polys_by_time

    if time_layers is not None:
        TIME_LAYERS = int(time_layers)

    TIME_SAMPLES = np.linspace(0.0, T_PERIOD, TIME_LAYERS, endpoint=False)
    WAIT_STEP = T_PERIOD / TIME_LAYERS

    camera_polys_by_time = {}
    for t in TIME_SAMPLES:
        polys = []
        for cam in cameras:
            poly = cam.visible_polygon(walls, resolution=CAMERA_RESOLUTION, t=t)
            polys.append(prep(poly))
        camera_polys_by_time[t] = polys


def configure_planner(nmax=None, dstep=None, time_layers=None, seed=None):
    global NMAX, DSTEP

    if nmax is not None:
        NMAX = int(nmax)
    if dstep is not None:
        DSTEP = float(dstep)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    rebuild_time_layers(time_layers=time_layers)

############################################################
# PRECOMPUTED CAMERA POLYGONS
############################################################

rebuild_time_layers()


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
    
    def distance(self, other):
        return sqrt((other.x - self.x)**2 + (other.y - self.y)**2)
    
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y), 0.0)
    
    def connectsTo(self, other):
        line = LineString([(self.x, self.y), (other.x, other.y)])
        return obstacles.disjoint(line)

    def inSpatialFreespace(self):
        return obstacles.disjoint(Point(self.x, self.y))


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
# RRT FUNCTIONS
############################################################



def rrt(startnode, goalnode, visual=None):
    startnode.parent = None
    tree = [startnode]

    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if visual:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            visual.show()

    def sample_node():
        if random.random() < GOAL_BIAS:
            return Node(goalnode.x, goalnode.y, 0.0)
        return Node(random.uniform(xmin, xmax), random.uniform(ymin, ymax), 0.0)

    steps = 0
    while True:
        q_rand = sample_node()
        q_near = min(tree, key=lambda node: node.distance(q_rand))

        distance = q_near.distance(q_rand)
        if distance > 0:
            alpha = min(DSTEP / distance, 1)
            q_new = q_near.intermediate(q_rand, alpha)

            if already_in_tree(tree, q_new):
                steps += 1
                if (steps >= SMAX) or (len(tree) >= NMAX):
                    print("Aborted after %d steps and the tree having %d nodes" %
                            (steps, len(tree)))
                    return None
                continue

            if q_new.inSpatialFreespace() and q_near.connectsTo(q_new):
                addtotree(q_near, q_new)

                if q_new.distance(goalnode) <= DSTEP and q_new.connectsTo(goalnode):
                    addtotree(q_new, goalnode)
                    break

        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" %
                    (steps, len(tree)))
            return None

    # Build the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Report and return.
    print("Finished after %d steps and the tree having %d nodes" %
            (steps, len(tree)))
    return path


# Compute the path cost
def pathCost(path):
    cost = 0
    for i in range(1, len(path)):
        cost += path[i-1].distance(path[i])
    return cost

# Post process the path
def postProcess(path):
    shortpath = [path[0]]
    for i in range(2, len(path)):
        if not shortpath[-1].connectsTo(path[i]):
            shortpath.append(path[i-1])
    shortpath.append(path[-1])
    return shortpath 

def optimize_stealth(path):
    """
    Assign times to a geometric path by searching over discrete wait actions.
    This keeps the spatial path fixed and uses the temporal logic from the
    XYT planner only in post-processing.
    """
    print("Optimizing path for stealth (searching over wait layers)...")
    if not path:
        return None

    start = Node(path[0].x, path[0].y, 0.0)
    if not start.inFreespace():
        print("Start node is visible at t = 0.0, so no valid timed path exists.")
        return None

    states = {0.0: (0.0, [start])}

    for i in range(len(path) - 1):
        next_states = {}

        for _, (elapsed, timed_prefix) in states.items():
            curr_node = timed_prefix[-1]
            next_spatial = path[i + 1]
            travel_time = curr_node.spatialDistance(next_spatial) / ROBOT_SPEED

            for wait_steps in range(TIME_LAYERS):
                wait_time = wait_steps * WAIT_STEP
                depart_t = wrap_time(curr_node.t + wait_time)
                depart_node = Node(curr_node.x, curr_node.y, depart_t)

                if wait_steps > 0 and not curr_node.edgeIsSafe(depart_node):
                    continue

                arrival_t = wrap_time(depart_t + travel_time)
                arrival_node = Node(next_spatial.x, next_spatial.y, arrival_t)

                if not arrival_node.inFreespace():
                    continue

                if not depart_node.edgeIsSafe(arrival_node):
                    continue

                candidate_path = list(timed_prefix)
                if wait_steps > 0:
                    candidate_path.append(depart_node)
                candidate_path.append(arrival_node)

                candidate_cost = elapsed + wait_time + travel_time
                key = round(arrival_t, 6)
                best = next_states.get(key)
                if best is None or candidate_cost < best[0]:
                    next_states[key] = (candidate_cost, candidate_path)

        if not next_states:
            print(f"No safe temporal schedule found for segment {i}.")
            return None

        states = next_states

    _, best_path = min(states.values(), key=lambda item: item[0])
    return best_path
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

def run_planner(show_visual=True, animate=False):
    result = {
        "success_spatial": False,
        "success_temporal": False,
        "spatial_cost": None,
        "temporal_cost": None,
        "spatial_nodes": 0,
        "temporal_nodes": 0,
        "timed_path_duration": None,
        "path": None,
        "finalpath": None,
    }

    print("Running 3D temporal PRM")
    print("Robot speed =", ROBOT_SPEED)
    print("Camera omega =", CAMERA_OMEGA)
    print("Time period =", T_PERIOD)
    print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

    visual = None
    if show_visual:
        visual = Visualization()
        visual.show()

    startnode = Node(xstart, ystart, 0.0)
    goalnode  = Node(xgoal,  ygoal, 0.0)

    if visual:
        visual.drawNode(startnode, color='orange', marker='o')
        visual.drawNode(goalnode,  color='purple', marker='o')
        visual.show("Showing basic world")

    print("Running RRT...")
    path = rrt(startnode, goalnode, visual)
    result["path"] = path

    if not path:
        if visual:
            visual.show("UNABLE TO FIND A PATH")
        return result

    result["success_spatial"] = True
    result["spatial_cost"] = pathCost(path)
    result["spatial_nodes"] = len(path)

    if visual:
        visual.drawPath(path, color='r', linewidth=2)
        visual.show("Showing the raw path (cost/length %.1f)" % result["spatial_cost"])

    finalpath = postProcess(path)

    finalpath = optimize_stealth(finalpath)
    result["finalpath"] = finalpath
    if not finalpath:
        if visual:
            visual.show("UNABLE TO FIND A SAFE TEMPORAL SCHEDULE")
        return result

    result["success_temporal"] = True
    result["temporal_cost"] = pathCost(finalpath)
    result["temporal_nodes"] = len(finalpath)
    result["timed_path_duration"] = sum(
        finalpath[i - 1].dt_forward(finalpath[i]) for i in range(1, len(finalpath))
    )

    if visual:
        visual.drawPath(finalpath, color='b', linewidth=2)
        visual.show("Showing the post-processed path (cost/length %.1f)" % result["temporal_cost"])

    if animate:
        animate_path(finalpath, frames_per_edge=25, interval=80)

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmax", type=int, default=NMAX)
    parser.add_argument("--dstep", type=float, default=DSTEP)
    parser.add_argument("--time-layers", type=int, default=TIME_LAYERS)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--no-animate", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_planner(
        nmax=args.nmax,
        dstep=args.dstep,
        time_layers=args.time_layers,
        seed=args.seed,
    )

    animate = args.animate or not args.no_viz
    if args.no_animate:
        animate = False

    t0 = time.time()
    result = run_planner(show_visual=not args.no_viz, animate=animate)
    runtime = time.time() - t0

    summary = {
        "nmax": NMAX,
        "dstep": DSTEP,
        "time_layers": TIME_LAYERS,
        "seed": args.seed,
        "success_spatial": result["success_spatial"],
        "success_temporal": result["success_temporal"],
        "spatial_cost": result["spatial_cost"],
        "temporal_cost": result["temporal_cost"],
        "spatial_nodes": result["spatial_nodes"],
        "temporal_nodes": result["temporal_nodes"],
        "timed_path_duration": result["timed_path_duration"],
        "runtime_sec": runtime,
    }

    if args.json:
        print(json.dumps(summary))
    else:
        print(summary)

    # print("Running A*...")
    # path = astar(nodes, startnode, goalnodes)

    # if not path:
    #     print("NO PATH FOUND")
    #     return

    # print("PATH FOUND")
    # print("Elapsed time cost:", path[-1].creach)
    # print("Arrival phase:", path[-1].t)

    # for p in path:
    #     print(p)

    # visual.drawPath(path, color='blue', linewidth=3)
    # visual.show("Temporal stealth path found")


if __name__ == "__main__":
    main()
