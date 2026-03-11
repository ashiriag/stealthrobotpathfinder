import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from math import cos, sin, pi
import random

cell_size = 1


# ============================================================
# Maze generation
# ============================================================

def generate_maze(rows, cols):
    visited = [[False] * cols for _ in range(rows)]
    walls = []

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def carve(r, c):
        visited[r][c] = True
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and not visited[nr][nc]:
                if dr == 0 and dc == 1:
                    x0 = c * cell_size + cell_size
                    y0 = r * cell_size
                    walls.append(
                        Polygon([
                            (x0, y0),
                            (x0 + 0.1, y0),
                            (x0 + 0.1, y0 + cell_size),
                            (x0, y0 + cell_size)
                        ])
                    )
                elif dr == 1 and dc == 0:
                    x0 = c * cell_size
                    y0 = r * cell_size + cell_size
                    walls.append(
                        Polygon([
                            (x0, y0),
                            (x0 + cell_size, y0),
                            (x0 + cell_size, y0 + 0.1),
                            (x0, y0 + 0.1)
                        ])
                    )
                carve(nr, nc)

    carve(0, 0)
    return walls


# ============================================================
# Camera class with time-varying direction
# ============================================================

class Camera:
    def __init__(self, x, y, direction, fov_angle, max_range, omega=0.0):
        """
        Parameters
        ----------
        x, y : float
            Camera position
        direction : float
            Initial direction at t = 0 (radians)
        fov_angle : float
            Field-of-view angle (radians)
        max_range : float
            Maximum sensing range
        omega : float
            Angular velocity in rad/s
            Positive = CCW, negative = CW
        """
        self.x = x
        self.y = y
        self.direction = direction        # keep for compatibility
        self.direction0 = direction       # true initial direction
        self.fov_angle = fov_angle
        self.max_range = max_range
        self.omega = omega

    def direction_at(self, t):
        """
        Camera viewing direction at time t.
        """
        return self.direction0 + self.omega * t

    def visible_polygon(self, walls, resolution=360, t=None):
        """
        Compute visible polygon at time t.
        If t is None, uses the current stored direction (backward compatible).
        If t is given, uses direction_at(t).
        """
        wall_lines = [LineString(w.exterior.coords) for w in walls]
        rays = []

        if t is None:
            direction = self.direction
        else:
            direction = self.direction_at(t)

        angles = np.linspace(
            direction - self.fov_angle / 2,
            direction + self.fov_angle / 2,
            resolution
        )

        epsilon = 0.05

        for a in angles:
            x_end = self.x + self.max_range * cos(a)
            y_end = self.y + self.max_range * sin(a)
            ray = LineString([(self.x, self.y), (x_end, y_end)])

            closest_pt = (x_end, y_end)
            min_dist = self.max_range

            for wall_line in wall_lines:
                inter = ray.intersection(wall_line)

                if not inter.is_empty:
                    if inter.geom_type == "Point":
                        d = np.hypot(inter.x - self.x, inter.y - self.y)
                        if epsilon < d < min_dist:
                            closest_pt = (inter.x, inter.y)
                            min_dist = d

                    elif hasattr(inter, "geoms"):
                        for pt in inter.geoms:
                            if pt.geom_type == "Point":
                                d = np.hypot(pt.x - self.x, pt.y - self.y)
                                if epsilon < d < min_dist:
                                    closest_pt = (pt.x, pt.y)
                                    min_dist = d

            rays.append(closest_pt)

        return Polygon([(self.x, self.y)] + rays).buffer(0)

    def auto_orient(self, walls, min_area=0.5, angle_step=pi / 18):
        """
        Finds a good initial orientation at t = 0.
        This sets both direction and direction0.
        """
        best_area = 0
        best_dir = self.direction0

        for k in range(int(2 * pi / angle_step)):
            test_dir = k * angle_step
            self.direction = test_dir
            poly = self.visible_polygon(walls, t=None)
            area = poly.area

            if area > best_area:
                best_area = area
                best_dir = test_dir

            if area > min_area:
                break

        self.direction = best_dir
        self.direction0 = best_dir

    def set_initial_direction(self, direction):
        """
        Explicitly reset the initial direction.
        """
        self.direction = direction
        self.direction0 = direction


# ============================================================
# Optional helper functions for temporal planning
# ============================================================

def point_in_camera_fov(point_xy, cam, walls, t, resolution=180):
    """
    Simple helper: returns True if point is inside the visible polygon
    of camera cam at time t.
    """
    poly = cam.visible_polygon(walls, resolution=resolution, t=t)
    from shapely.geometry import Point
    return poly.contains(Point(point_xy[0], point_xy[1]))


def cameras_visible_polygons(cameras, walls, t, resolution=180):
    """
    Return all camera visible polygons at time t.
    """
    return [cam.visible_polygon(walls, resolution=resolution, t=t) for cam in cameras]


# ============================================================
# Drawing
# ============================================================

def draw_maze(cameras, walls, start, goal, rows, cols, t=0.0):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw walls
    for wall in walls:
        x, y = wall.exterior.xy
        ax.fill(x, y, color='black')

    # Draw cameras at time t
    for cam in cameras:
        poly = cam.visible_polygon(walls, t=t)
        x, y = poly.exterior.xy
        ax.fill(x, y, color='red', alpha=0.3)
        ax.plot(cam.x, cam.y, 'ro')

        # draw a small heading line to show orientation
        theta = cam.direction_at(t)
        hx = cam.x + 0.4 * cos(theta)
        hy = cam.y + 0.4 * sin(theta)
        ax.plot([cam.x, hx], [cam.y, hy], 'r-', linewidth=2)

    # Draw start & goal
    ax.plot(start[0], start[1], 'go', markersize=10, label="Start")
    ax.plot(goal[0], goal[1], 'bo', markersize=10, label="Goal")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(f"Stealth Maze with Camera FOV at t = {t:.2f}")
    ax.legend()
    plt.gca().invert_yaxis()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    random.seed(9)
    np.random.seed(9)

    rows = 12
    cols = 10

    walls = generate_maze(rows, cols)

    CAMERA_OMEGA = pi / 6   # rad/s

    cameras = [
        Camera(5, 1, direction=pi, fov_angle=pi/2, max_range=5, omega=CAMERA_OMEGA),
        Camera(3, 8, direction=pi, fov_angle=pi/2, max_range=5, omega=CAMERA_OMEGA),
        Camera(9, 6, direction=pi, fov_angle=pi/2, max_range=5, omega=CAMERA_OMEGA),
    ]

    for cam in cameras:
        cam.auto_orient(walls)

    start = (0.5, 0.5)
    goal = (cols - 0.5, rows - 0.5)

    # show world at t = 0
    draw_maze(cameras, walls, start, goal, rows, cols, t=0.0)

    # show world at a later time
    draw_maze(cameras, walls, start, goal, rows, cols, t=2.0)