import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt

EPSILON = 0.0001  # the smallest value we consider a number
INF = 1000000.0  # the largest value we consider a number

MAX_DEPTH = 4  # max ray bounces
RENDER_WIDTH = 64
RENDER_HEIGHT = 64
SPP = 5

scene = []


def normalize(vector):
    """
    returns a normalized unit vector in the same direction. For some reason numpy did not have this already.
    :param vector:
    :return:
    """
    return vector / la.norm(vector)

# =================================================
# OBJECTS
# Ray object


class Ray:
    # Initializer
    def __init__(self, origin=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 0.0])):
        self.origin = origin
        self.dir = direction

    # Gets the actual hit coordinates from distance
    def get_hit_point(self, t):
        return self.origin + (self.dir * t)


# Sphere object
class Sphere:
    def __init__(self, sphere_origin, sphere_radius, diff, emit):
        self.origin = sphere_origin
        self.radius = sphere_radius
        self.diff = diff
        self.emit = emit

    # Tests for intersection between object and ray.
    # returns the distance and the normal of the exact hit point.
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    def intersect(self, ray):
        rad2 = self.radius * self.radius

        length = self.origin - ray.origin
        tca = np.dot(length, ray.dir)

        if tca < 0.0:
            return -1.0, None
        d2 = (np.dot(length, length)) - (tca*tca)

        if d2 > rad2:
            return -1.0, None
        thc = np.sqrt(rad2 - d2)

        t0 = tca - thc  # we're only really interested in the nearest (first) hit point.

        hit_point = ray.get_hit_point(t0)
        normal = normalize(hit_point - self.origin)

        return t0, normal


# Plane object
class Plane:
    def __init__(self, plane_origin, plane_normal, diff, emit):
        self.origin = plane_origin
        self.normal = normalize(plane_normal)
        self.diff = diff
        self.emit = emit

    # Tests for intersection between object and ray.
    # returns the distance and the normal of the exact hit point.
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
    def intersect(self, ray):
        denominator = np.dot(self.normal, ray.dir)

        if abs(denominator) > EPSILON:
            t = np.dot(self.normal, (self.origin - ray.origin)) / denominator
            if t >= 0:
                return t, self.normal

        return -1.0, None


class Camera:
    # Initializer
    def __init__(self, orig, view_plane_dist, view_plane_width, view_plane_height):
        self.origin = orig
        self.view_plane_dist = view_plane_dist
        self.view_plane_height = view_plane_width
        self.view_plane_width = view_plane_height

        # setup orthonormal basis
        self.u = np.array([1.0, 0.0, 0.0])
        self.v = np.array([0.0, 1.0, 0.0])
        self.w = np.array([0.0, 0.0, 1.0])

    # we determine the initial ray direction by simply adding a bit of left-ness and a bit of top-ness
    # to our camera's forward direction (depending on the pixel coordinates)
    # then re-normalizing to get a direction out of it.
    def orient_ray(self, x, y):
        view_space_x = (x - (RENDER_WIDTH / 2.0)) * (self.view_plane_width / RENDER_WIDTH)
        view_space_y = (y - (RENDER_HEIGHT / 2.0)) * (self.view_plane_height / RENDER_HEIGHT)
        direction = (self.u * view_space_x) + (self.v * view_space_y) + (self.w * self.view_plane_dist)
        return normalize(direction)

    # Gather SPP amount of samples for each pixel
    def simulate(self):
        pixel_samples = np.empty((RENDER_WIDTH, RENDER_HEIGHT, SPP, 3))
        ray = Ray()
        ray.origin = self.origin
        for x in range(0, RENDER_WIDTH):
            for y in range(0, RENDER_HEIGHT):
                for s in range(0, SPP):
                    # For a given pixel "coordinate", we actually want the integral over the rectangle of that pixel
                    # We do this by randomly sampling over that area rectangle
                    u1 = np.random.random_sample() - 0.5
                    u2 = np.random.random_sample() - 0.5
                    sp_x = x + u1
                    sp_y = y + u2
                    ray.dir = self.orient_ray(sp_x, sp_y)
                    pixel_sample = trace_path(ray, 1)
                    pixel_samples[x, y, s, ] = pixel_sample
            print(x, "/", RENDER_WIDTH)
        return pixel_samples


def trace_path(ray, depth):
    return
