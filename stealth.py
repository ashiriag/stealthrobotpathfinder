import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from math import cos, sin, pi
import random

# ============================================================
# Maze Generation (Recursive Backtracking)
# ============================================================
cell_size = 1
def generate_maze(rows, cols):
    visited = [[False]*cols for _ in range(rows)]
    walls = []

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def carve(r, c):
        visited[r][c] = True
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if in_bounds(nr,nc) and not visited[nr][nc]:
                # Add wall between cells as rectangles
                if dr == 0 and dc == 1:
                    x0 = c*cell_size + cell_size
                    y0 = r*cell_size
                    walls.append(Polygon([(x0, y0), (x0+0.1, y0), (x0+0.1, y0+cell_size), (x0, y0+cell_size)]))
                elif dr == 1 and dc == 0:
                    x0 = c*cell_size
                    y0 = r*cell_size + cell_size
                    walls.append(Polygon([(x0, y0), (x0+cell_size, y0), (x0+cell_size, y0+0.1), (x0, y0+0.1)]))
                carve(nr,nc)

    carve(0,0)
    return walls
# ============================================================
# Camera Class
# ============================================================

class Camera:
    def __init__(self, x, y, direction, fov_angle, max_range):
        self.x = x
        self.y = y
        self.direction = direction
        self.fov_angle = fov_angle
        self.max_range = max_range

    def visible_polygon(self, walls, resolution=120):
        wall_lines = [LineString(w.exterior.coords) for w in walls]
        rays = []

        angles = np.linspace(
            self.direction - self.fov_angle/2,
            self.direction + self.fov_angle/2,
            resolution
        )

        epsilon = 0.05

        for a in angles:
            x_end = self.x + self.max_range * cos(a)
            y_end = self.y + self.max_range * sin(a)
            ray = LineString([(self.x, self.y), (x_end, y_end)])

            closest_pt = (self.x + epsilon*cos(a), self.y + epsilon*sin(a))
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

        return Polygon([(self.x, self.y)] + rays)
    def auto_orient(self, walls, min_area=0.5, angle_step=pi/18):
        best_area = 0
        best_dir = self.direction

        for k in range(int(2*pi / angle_step)):
            test_dir = k * angle_step
            self.direction = test_dir
            poly = self.visible_polygon(walls)
            area = poly.area

            if area > best_area:
                best_area = area
                best_dir = test_dir

            if area > min_area:
                break

        self.direction = best_dir

# ============================================================
# Drawing Function
# ============================================================

def draw_maze(cameras, walls, start, goal, rows, cols):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw walls
    for wall in walls:
        x, y = wall.exterior.xy
        ax.fill(x, y, color='black')

    # Draw cameras
    for cam in cameras:
        poly = cam.visible_polygon(walls)
        x, y = poly.exterior.xy
        ax.fill(x, y, color='red', alpha=0.3)
        ax.plot(cam.x, cam.y, 'ro')

    # Draw start & goal
    ax.plot(start[0], start[1], 'go', markersize=10, label="Start")
    ax.plot(goal[0], goal[1], 'bo', markersize=10, label="Goal")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title("Stealth Maze with Camera FOV")
    ax.legend()
    plt.gca().invert_yaxis()
    plt.show()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    rows = 12
    cols = 10

    walls = generate_maze(rows, cols)

    cameras = [
        Camera(1.5, 1.5, direction=pi/4, fov_angle=pi/3, max_range=4),
        Camera(8.5, 10, direction=-pi/2, fov_angle=pi/4, max_range=4),
        Camera(5, 6, direction=pi, fov_angle=pi/2, max_range=3)
    ]

    # Auto-rotate cameras if blocked
    for cam in cameras:
        cam.auto_orient(walls)

    start = (0.5, 0.5)
    goal  = (cols - 0.5, rows - 0.5)

    draw_maze(cameras, walls, start, goal, rows, cols)