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


random.seed(9)
np.random.seed(9)

# Period of one full camera rotation
T_PERIOD = 2 * pi / abs(CAMERA_OMEGA)
TIME_SAMPLES = np.linspace(0.0, T_PERIOD, TIME_LAYERS, endpoint=False)

WAIT_STEP = T_PERIOD / TIME_LAYERS   # one time layer
GOAL_BIAS = 0.08
WAIT_PROB = 0.25                     # chance to try a wait expansion
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

def phase_distance(t1, t2):
    """
    Smallest circular difference between two phases in [0, T_PERIOD).
    """
    a = wrap_time(t1)
    b = wrap_time(t2)
    return min((a - b) % T_PERIOD, (b - a) % T_PERIOD)

def already_in_tree(tree, node):
    for n in tree:
        if n.spatialDistance(node) <= MERGE_POS_EPS and phase_distance(n.t, node.t) <= MERGE_TIME_EPS:
            return True
    return False
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

    def stateDistance(self, other):
        """
        Distance in space-time for nearest-neighbor lookup.
        Convert phase difference into an equivalent spatial scale using robot speed.
        """
        ds = self.spatialDistance(other)
        dt = phase_distance(self.t, other.t)
        return sqrt(ds**2 + (ROBOT_SPEED * dt)**2)
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



def rrt_temporal(startnode, goal_xy, visual=None):
    """
    Space-time RRT.
    Nodes are (x, y, t).
    Expansions are either:
      1) motion edges
      2) wait edges
    """
    startnode.parent = None
    tree = [startnode]

    def addtotree(oldnode, newnode, color='g'):
        newnode.parent = oldnode
        tree.append(newnode)
        if visual:
            visual.drawEdge(oldnode, newnode, color=color, linewidth=1)
            visual.show()

    def sample_state():
        # Goal bias: sample the goal location with a random time phase
        if random.random() < GOAL_BIAS:
            return Node(goal_xy[0], goal_xy[1], random.choice(TIME_SAMPLES))

        return Node(
            random.uniform(xmin, xmax),
            random.uniform(ymin, ymax),
            random.choice(TIME_SAMPLES)
        )

    def nearest_node(q_rand):
        return min(tree, key=lambda node: node.stateDistance(q_rand))

    def steer_motion(q_near, q_rand):
        """
        Step spatially toward q_rand. Time is determined by travel time.
        """
        d = q_near.spatialDistance(q_rand)
        if d < 1e-9:
            return None

        alpha = min(DSTEP / d, 1.0)

        x_new = q_near.x + alpha * (q_rand.x - q_near.x)
        y_new = q_near.y + alpha * (q_rand.y - q_near.y)

        d_step = sqrt((x_new - q_near.x)**2 + (y_new - q_near.y)**2)
        t_new = wrap_time(q_near.t + d_step / ROBOT_SPEED)

        return Node(x_new, y_new, t_new)

    def steer_wait(q_near, q_rand=None):
        """
        Stay in place and advance time.
        Optionally bias toward the sampled phase if q_rand is provided.
        """
        if q_rand is None:
            dt = WAIT_STEP
        else:
            desired = q_near.dt_forward(q_rand)
            if desired < 1e-6:
                dt = WAIT_STEP
            else:
                dt = min(desired, WAIT_STEP)

        return Node(q_near.x, q_near.y, wrap_time(q_near.t + dt))

    def can_connect_to_goal(q_new):
        """
        Try connecting directly to the goal position.
        The goal time is determined by travel time from q_new.
        """
        gx, gy = goal_xy
        goal_spatial = Node(gx, gy, 0.0)

        d = q_new.spatialDistance(goal_spatial)
        if d > DSTEP:
            return None

        t_goal = wrap_time(q_new.t + d / ROBOT_SPEED)
        q_goal = Node(gx, gy, t_goal)

        if q_goal.inFreespace() and q_new.edgeIsSafe(q_goal):
            return q_goal
        return None

    steps = 0
    while True:
        q_rand = sample_state()
        q_near = nearest_node(q_rand)

        candidates = []

        # Motion candidate
        q_move = steer_motion(q_near, q_rand)
        if q_move is not None:
            candidates.append(("move", q_move))

        # Wait candidate
        # Either sometimes explore waits randomly, or when sampled target is near in space
        if random.random() < WAIT_PROB or q_near.spatialDistance(q_rand) < DSTEP:
            q_wait = steer_wait(q_near, q_rand)
            candidates.append(("wait", q_wait))

        added = False
        for kind, q_new in candidates:
            if already_in_tree(tree, q_new):
                continue

            if not q_new.inFreespace():
                continue

            if not q_near.edgeIsSafe(q_new):
                continue

            addtotree(q_near, q_new, color=('c' if kind == "wait" else 'g'))
            added = True

            q_goal = can_connect_to_goal(q_new)
            if q_goal is not None and not already_in_tree(tree, q_goal):
                addtotree(q_new, q_goal, color='m')

                path = [q_goal]
                while path[0].parent is not None:
                    path.insert(0, path[0].parent)

                print("Finished after %d steps and the tree having %d nodes" %
                      (steps, len(tree)))
                return path

            break  # only add one node per iteration

        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" %
                  (steps, len(tree)))
            return None
        
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
    Post-processes the path to insert explicit 'Wait' nodes.
    If a segment is unsafe, the robot stays at the current (x,y)
    until the cameras have rotated to a safe configuration.
    """
    print("Optimizing path for stealth (inserting wait behaviors)...")
    # Initialize with the start node at t=0
    path[0].t = 0.0
    new_path = [path[0]]
    
    for i in range(len(path) - 1):
        curr_node = new_path[-1]
        next_spatial_node = path[i+1]
        
        dist = curr_node.spatialDistance(next_spatial_node)
        travel_time = dist / ROBOT_SPEED
        
        wait_time = 0.0
        step_size = 0.2  # Check every 0.2 seconds for an opening
        max_wait = T_PERIOD
        
        found_window = False
        while wait_time < max_wait:
            # We simulate a "Wait" at current position followed by "Motion" to next position
            arrival_time = wrap_time(curr_node.t + wait_time + travel_time)
            temp_next = Node(next_spatial_node.x, next_spatial_node.y, arrival_time)
            
            # Check if the motion edge from (curr.x, curr.y, curr.t + wait) 
            # to (next.x, next.y, arrival_t) is safe.
            # We temporarily update curr_node.t to check safety for the motion part
            original_t = curr_node.t
            curr_node.t = wrap_time(original_t + wait_time)
            
            if curr_node.edgeIsSafe(temp_next):
                # If we had to wait, add a node representing the end of the wait
                if wait_time > 0:
                    wait_node = Node(curr_node.x, curr_node.y, curr_node.t)
                    wait_node.parent = new_path[-1]
                    new_path.append(wait_node)
                
                # Add the actual movement node
                temp_next.parent = new_path[-1]
                new_path.append(temp_next)
                found_window = True
                break
            
            # Reset t and try waiting longer
            curr_node.t = original_t
            wait_time += step_size
            
        if not found_window:
            print(f"Warning: No safe window found for segment {i}. Moving anyway.")
            arrival_time = wrap_time(curr_node.t + travel_time)
            next_spatial_node.t = arrival_time
            next_spatial_node.parent = curr_node
            new_path.append(next_spatial_node)

    return new_path

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
    # print("N =", N, "K =", K)
    print("Robot speed =", ROBOT_SPEED)
    print("Camera omega =", CAMERA_OMEGA)
    print("Time period =", T_PERIOD)

    visual = Visualization(show_camera_time=0.0)

    print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

    # Create the figure.  Some computers seem to need an additional show()?
    visual = Visualization()
    visual.show()

    # Create the start/goal nodes.
    startnode = Node(xstart, ystart, 0.0)
    goalnode  = Node(xgoal,  ygoal, 0.0)

    # Show the start/goal nodes.
    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode,  color='purple', marker='o')
    visual.show("Showing basic world")


    # Run the RRT planner.
    print("Running RRT...")
    path = rrt_temporal(startnode, (xgoal, ygoal), visual)  
    # If unable to connect, just note before closing.
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    # Show the path.
    cost = pathCost(path)
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing the raw path (cost/length %.1f)" % cost)


    # # Post process the path.
    # finalpath = postProcess(path)

    # # Show the post-processed path.
    # cost = pathCost(finalpath)
    # # finalpath = optimize_stealth(finalpath)
    # visual.drawPath(finalpath, color='b', linewidth=2)
    # visual.show("Showing the post-processed path (cost/length %.1f)" % cost)
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    visual.drawPath(path, color='b', linewidth=2)
    visual.show("Showing temporal path")

    animate_path(path, frames_per_edge=25, interval=80)
    # Set the starting time
    # finalpath[0].t = 0.0
    
    # Iterate through the path to update time stamps based on ROBOT_SPEED
    # for i in range(len(finalpath) - 1):
    #     current_node = finalpath[i]
    #     next_node = finalpath[i+1]
        
    #     # Calculate time taken to travel the distance
    #     travel_time = current_node.distance(next_node) / ROBOT_SPEED
        
    #     # Set the next node's time (wrapped to the camera period)
    #     next_node.t = wrap_time(current_node.t + travel_time)



    # startnode = Node(xstart, ystart, 0.0)

    # goalnodes = addGoalNodes(nodes, num_goal_times=TIME_LAYERS)

    # if not goalnodes:
    #     print("No valid goal-time nodes found")
    #     return

    # for g in goalnodes:
    #     visual.drawNode(g, color='purple', marker='o', markersize=4)

    # if not startnode.inFreespace():
    #     print("Start node is invalid at t = 0")
    #     return

    # visual.drawNode(startnode, color='orange', marker='o', markersize=8)
    # visual.show("World ready")

    # print("Sampling 3D nodes...")
    # nodes = createNodes(N)

    # for node in nodes:
        # visual.drawNode(node, color='k', marker='x', markersize=3)

    # nodes.append(startnode)

    # goalnodes = addGoalNodes(nodes, num_goal_times=TIME_LAYERS)

    # if not goalnodes:
    #     print("No valid goal-time nodes found")
    #     return

    # for g in goalnodes:
    #     visual.drawNode(g, color='purple', marker='o', markersize=4)

    # visual.show("Nodes sampled")

    # print("Connecting neighbors...")
    # connectNearestNeighbors(nodes, K)



    # edge_count = 0
    # for i, node in enumerate(nodes):
    #     for neighbor in node.neighbors:
    #         edge_count += 1
    #         visual.drawEdge(node, neighbor, color='green', linewidth=0.2)

    # print("Directed edges:", edge_count)
    # visual.show("Graph built")

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

    # animate_path(finalpath, frames_per_edge=25, interval=80)


if __name__ == "__main__":
    main()
