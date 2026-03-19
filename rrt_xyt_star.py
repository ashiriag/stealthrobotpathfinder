"""
rrtstar_stealth_3d.py

RRT* for stealth planning with rotating cameras.
State space is (x, y, t).

- Robot moves at constant speed
- Cameras rotate at constant angular speed
- Time is determined physically by travel distance
- Waiting is inserted on-demand when a camera blocks the path
- Parent selection uses RRT* logic: pick parent with lowest creach + wait + travel
- Rewiring updates neighbors if a cheaper path is found through the new node
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
DSTEP = 1.0
NEAR_RADIUS = DSTEP * 2.5   # radius for RRT* parent selection / rewiring

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 50000
NMAX = 1500

(xmin, xmax) = (0, 10)
(ymin, ymax) = (0, 12)

rows = 12
cols = 10

ROBOT_SPEED = 1.0         # units / second
CAMERA_OMEGA = pi / 6     # rad / second
TIME_LAYERS = 40          # discrete time layers for camera polygon precomputation
WAIT_EPS = 0.20           # spatial threshold for wait edges
EDGE_CHECK_STEPS = 25     # interpolation samples per edge check
CAMERA_RESOLUTION = 120   # visible polygon resolution

random.seed(9)
np.random.seed(9)

# Period of one full camera rotation
T_PERIOD = 2 * pi / abs(CAMERA_OMEGA)
TIME_SAMPLES = np.linspace(0.0, T_PERIOD, TIME_LAYERS, endpoint=False)
WAIT_STEP = T_PERIOD / TIME_LAYERS

GOAL_BIAS = 0.08
MERGE_POS_EPS = 0.25
MERGE_TIME_EPS = WAIT_STEP / 2


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

for cam in cameras:
    cam.auto_orient(walls)

(xstart, ystart) = (0.5, 0.5)
(xgoal,  ygoal)  = (cols - 0.5, rows - 0.5)


############################################################
# HELPERS
############################################################

def wrap_time(t):
    return t % T_PERIOD


def nearest_time_sample(t):
    idx = int(np.argmin(np.abs(TIME_SAMPLES - wrap_time(t))))
    return TIME_SAMPLES[idx]


def point_seen_at_time(x, y, t):
    t_snap = nearest_time_sample(t)
    p = Point(x, y)
    for poly in camera_polys_by_time[t_snap]:
        if poly.contains(p):
            return True
    return False


def get_camera_polygons_at_time(t):
    polys = []
    for cam in cameras:
        poly = cam.visible_polygon(walls, resolution=CAMERA_RESOLUTION, t=t)
        polys.append(poly)
    return polys


def phase_distance(t1, t2):
    a = wrap_time(t1)
    b = wrap_time(t2)
    return min((a - b) % T_PERIOD, (b - a) % T_PERIOD)


def already_in_tree(tree, node):
    for n in tree:
        if n.spatialDistance(node) <= MERGE_POS_EPS and phase_distance(n.t, node.t) <= MERGE_TIME_EPS:
            return True
    return False


def interpolate_path(path, frames_per_edge=20):
    samples = []
    if not path:
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
    samples.append((path[-1].x, path[-1].y, path[-1].t))
    return samples


############################################################
# PRECOMPUTED CAMERA POLYGONS
############################################################

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

        for wall in walls:
            x, y = wall.exterior.xy
            plt.fill(x, y, color='black')

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
        for i in range(len(path) - 1):
            self.drawEdge(path[i], path[i + 1], **kwargs)


############################################################
# NODE
############################################################

class Node:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = wrap_time(t)
        self.parent = None
        self.creach = inf   # total elapsed time from start

    # --------------------------------------------------
    # distances
    # --------------------------------------------------

    def spatialDistance(self, other):
        return sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)

    def dt_forward(self, other):
        return (other.t - self.t) % T_PERIOD

    # --------------------------------------------------
    # validity
    # --------------------------------------------------

    def inFreespace(self):
        p = Point(self.x, self.y)
        if not obstacles.disjoint(p):
            return False
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
        # allow a tolerance of up to one WAIT_STEP so that inserted
        # wait nodes don't break the feasibility check
        return abs(dt - required) <= WAIT_STEP + 1e-6

    def is_dynamically_feasible(self, other):
        return self.is_wait_edge(other) or self.is_motion_edge(other)

    def edgeIsSafe(self, other):
        if not self.is_dynamically_feasible(other):
            return False

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
            if not obstacles.disjoint(Point(x, y)):
                return False
            if point_seen_at_time(x, y, t):
                return False
        return True

    def __repr__(self):
        return f"<Node {self.x:.2f},{self.y:.2f},t={self.t:.2f},c={self.creach:.2f}>"


############################################################
# COST HELPER
############################################################

def cost_through(q_parent, x_new, y_new):
    """
    Compute the cheapest way to reach (x_new, y_new) from q_parent.

    Scans forward in time from q_parent.t, trying departures spaced
    by WAIT_STEP, until the motion edge is safe or a full period
    has elapsed.

    Returns (total_edge_cost, wait_time, t_arrive) or (inf, None, None).
      total_edge_cost = wait_time + travel_time
    """
    d = sqrt((x_new - q_parent.x) ** 2 + (y_new - q_parent.y) ** 2)
    if d < 1e-9:
        return inf, None, None

    travel_time = d / ROBOT_SPEED

    wait = 0.0
    while wait < T_PERIOD + 1e-9:
        t_depart = wrap_time(q_parent.t + wait)
        t_arrive = wrap_time(t_depart + travel_time)

        depart_node = Node(q_parent.x, q_parent.y, t_depart)
        arrive_node = Node(x_new, y_new, t_arrive)

        if depart_node.edgeIsSafe(arrive_node):
            return wait + travel_time, wait, t_arrive

        wait += WAIT_STEP

    return inf, None, None


############################################################
# RRT* TEMPORAL
############################################################

def rrtstar_temporal(startnode, goal_xy, visual=None):
    """
    RRT* in (x, y, t) space with on-demand waiting.

    For each sampled (x, y):
      - Find all nearby tree nodes within NEAR_RADIUS
      - For each, compute cost_through: how long to wait + travel
      - Pick the parent with lowest creach + edge_cost
      - Insert a wait node first if needed, then the new node
      - Rewire: check if existing nearby nodes are cheaper via the new node
    """
    startnode.creach = 0.0
    startnode.parent = None
    tree = [startnode]

    def addtotree(parent, node, color='g'):
        node.parent = parent
        tree.append(node)
        if visual:
            visual.drawEdge(parent, node, color=color, linewidth=1)
            visual.show()

    def sample_xy():
        if random.random() < GOAL_BIAS:
            return goal_xy[0], goal_xy[1]
        return random.uniform(xmin, xmax), random.uniform(ymin, ymax)

    def near_nodes(x, y):
        dummy = Node(x, y, 0.0)
        return [n for n in tree if n.spatialDistance(dummy) <= NEAR_RADIUS]

    def try_connect_goal(q_new):
        """
        If q_new is within DSTEP of goal, try to connect.
        Uses cost_through to find wait + arrival time.
        """
        gx, gy = goal_xy
        d = sqrt((gx - q_new.x) ** 2 + (gy - q_new.y) ** 2)
        if d > DSTEP:
            return None

        edge_cost, wait, t_arrive = cost_through(q_new, gx, gy)
        if edge_cost == inf:
            return None

        # insert wait node if needed
        if wait > WAIT_STEP / 2:
            t_depart = wrap_time(q_new.t + wait)
            wait_node = Node(q_new.x, q_new.y, t_depart)
            wait_node.creach = q_new.creach + wait
            if not wait_node.inFreespace():
                return None
            addtotree(q_new, wait_node, color='c')
            effective_parent = wait_node
        else:
            effective_parent = q_new

        q_goal = Node(gx, gy, t_arrive)
        q_goal.creach = effective_parent.creach + (edge_cost - wait)
        if not q_goal.inFreespace():
            return None
        addtotree(effective_parent, q_goal, color='m')
        return q_goal

    steps = 0
    best_goal = None

    while True:
        x_rand, y_rand = sample_xy()

        candidates = near_nodes(x_rand, y_rand)
        if not candidates:
            # fall back to spatially nearest
            dummy = Node(x_rand, y_rand, 0.0)
            candidates = [min(tree, key=lambda n: n.spatialDistance(dummy))]

        # --------------------------------------------------
        # Find the best parent for the new node at (x_rand, y_rand)
        # --------------------------------------------------
        best_parent = None
        best_total = inf
        best_wait = None
        best_t_arrive = None

        for q_near in candidates:
            edge_cost, wait, t_arrive = cost_through(q_near, x_rand, y_rand)
            if edge_cost == inf:
                continue
            total = q_near.creach + edge_cost
            if total < best_total:
                best_total = total
                best_parent = q_near
                best_wait = wait
                best_t_arrive = t_arrive

        if best_parent is None:
            steps += 1
            if steps >= SMAX or len(tree) >= NMAX:
                print("Aborted after %d steps, %d nodes" % (steps, len(tree)))
                return None
            continue

        # --------------------------------------------------
        # Insert wait node if the best parent needs to wait
        # --------------------------------------------------
        if best_wait > WAIT_STEP / 2:
            t_depart = wrap_time(best_parent.t + best_wait)
            wait_node = Node(best_parent.x, best_parent.y, t_depart)
            wait_node.creach = best_parent.creach + best_wait

            if already_in_tree(tree, wait_node) or not wait_node.inFreespace():
                steps += 1
                if steps >= SMAX or len(tree) >= NMAX:
                    print("Aborted after %d steps, %d nodes" % (steps, len(tree)))
                    return None
                continue

            addtotree(best_parent, wait_node, color='c')
            effective_parent = wait_node
            travel_time = sqrt((x_rand - best_parent.x)**2 + (y_rand - best_parent.y)**2) / ROBOT_SPEED
            node_creach = wait_node.creach + travel_time
        else:
            effective_parent = best_parent
            travel_time = sqrt((x_rand - best_parent.x)**2 + (y_rand - best_parent.y)**2) / ROBOT_SPEED
            node_creach = best_parent.creach + travel_time

        # --------------------------------------------------
        # Create and validate the new node
        # --------------------------------------------------
        q_new = Node(x_rand, y_rand, best_t_arrive)
        q_new.creach = node_creach

        if already_in_tree(tree, q_new) or not q_new.inFreespace():
            steps += 1
            if steps >= SMAX or len(tree) >= NMAX:
                print("Aborted after %d steps, %d nodes" % (steps, len(tree)))
                return None
            continue

        addtotree(effective_parent, q_new, color='g')

        # --------------------------------------------------
        # Rewire: check if nearby nodes are cheaper via q_new
        # --------------------------------------------------
        for q_neighbor in near_nodes(x_rand, y_rand):
            if q_neighbor is q_new or q_neighbor is effective_parent:
                continue

            edge_cost, wait, t_arrive = cost_through(q_new, q_neighbor.x, q_neighbor.y)
            if edge_cost == inf:
                continue

            new_cost = q_new.creach + edge_cost
            if new_cost < q_neighbor.creach:
                q_neighbor.parent = q_new
                q_neighbor.creach = new_cost
                q_neighbor.t = t_arrive

        # --------------------------------------------------
        # Try connecting to goal
        # --------------------------------------------------
        q_goal = try_connect_goal(q_new)
        if q_goal is not None:
            if best_goal is None or q_goal.creach < best_goal.creach:
                best_goal = q_goal
                print("Goal reached! creach=%.2f at step %d, nodes=%d" %
                      (best_goal.creach, steps, len(tree)))

                # Build and return path immediately (plain RRT* stops at first goal)
                path = [best_goal]
                while path[0].parent is not None:
                    path.insert(0, path[0].parent)
                print("Path has %d nodes" % len(path))
                return path

        steps += 1
        if steps >= SMAX or len(tree) >= NMAX:
            print("Aborted after %d steps, %d nodes" % (steps, len(tree)))
            if best_goal is not None:
                path = [best_goal]
                while path[0].parent is not None:
                    path.insert(0, path[0].parent)
                return path
            return None


############################################################
# PATH COST
############################################################

def pathCost(path):
    cost = 0
    for i in range(1, len(path)):
        dx = path[i].x - path[i-1].x
        dy = path[i].y - path[i-1].y
        cost += sqrt(dx*dx + dy*dy)
    return cost


############################################################
# ANIMATION
############################################################

def animate_path(path, frames_per_edge=20, interval=80, save=False, filename="stealth_animation.gif"):
    if not path:
        print("No path to animate.")
        return

    samples = interpolate_path(path, frames_per_edge=frames_per_edge)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Temporal Stealth RRT* Animation")

    for wall in walls:
        xw, yw = wall.exterior.xy
        ax.fill(xw, yw, color='black')

    ax.plot(xstart, ystart, 'go', markersize=10, label="Start")
    ax.plot(xgoal,  ygoal,  'bo', markersize=10, label="Goal")

    for i in range(len(path) - 1):
        ax.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y],
                color='deepskyblue', linewidth=2, alpha=0.5)

    robot_plot,  = ax.plot([], [], 'mo', markersize=8, label="Robot")
    trail_plot,  = ax.plot([], [], 'm-', linewidth=2, alpha=0.8)

    for cam in cameras:
        ax.plot(cam.x, cam.y, 'ro', markersize=6)

    cam_patches  = [ax.fill([], [], color='red', alpha=0.25)[0] for _ in cameras]
    cam_headings = [ax.plot([], [], 'r-', linewidth=2)[0]        for _ in cameras]

    time_text   = ax.text(0.02, 1.02, "", transform=ax.transAxes, fontsize=11)
    detect_text = ax.text(0.55, 1.02, "", transform=ax.transAxes, fontsize=11, color='darkred')

    xs_trail, ys_trail = [], []

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

        polys = get_camera_polygons_at_time(t)
        robot_seen = False

        for cam, poly, patch, hline in zip(cameras, polys, cam_patches, cam_headings):
            if not poly.is_empty and hasattr(poly, "exterior"):
                patch.set_xy(np.column_stack(poly.exterior.xy))
            else:
                patch.set_xy(np.empty((0, 2)))
            theta = cam.direction_at(t)
            hline.set_data([cam.x, cam.x + 0.5*np.cos(theta)],
                           [cam.y, cam.y + 0.5*np.sin(theta)])
            if poly.contains(Point(x, y)):
                robot_seen = True

        time_text.set_text(f"time = {t:.2f} s")
        detect_text.set_text("DETECTED" if robot_seen else "hidden")
        return [robot_plot, trail_plot, time_text, detect_text] + cam_patches + cam_headings

    anim = FuncAnimation(fig, update, frames=len(samples),
                         init_func=init, interval=interval,
                         blit=False, repeat=False)
    ax.legend()

    if save:
        anim.save(filename, dpi=120)
        print(f"Animation saved to {filename}")

    plt.show()


############################################################
# MAIN
############################################################

def main():
    print("Running Temporal RRT* Stealth Planner")
    print("Robot speed =", ROBOT_SPEED)
    print("Camera omega =", CAMERA_OMEGA)
    print("Time period =", T_PERIOD)
    print("Near radius =", NEAR_RADIUS)

    visual = Visualization(show_camera_time=0.0)
    visual = Visualization()
    visual.show()

    startnode = Node(xstart, ystart, 0.0)
    startnode.creach = 0.0
    goalnode  = Node(xgoal,  ygoal,  0.0)

    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode,  color='purple', marker='o')
    visual.show("Showing basic world")

    print("Running RRT*...")
    path = rrtstar_temporal(startnode, (xgoal, ygoal), visual)

    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    cost = pathCost(path)
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Path found (spatial length %.1f, elapsed time %.1f s)" %
                (cost, path[-1].creach))

    animate_path(path, frames_per_edge=25, interval=80)


if __name__ == "__main__":
    main()