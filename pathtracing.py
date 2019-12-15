import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt

EPSILON = 0.0001  # the smallest value we consider a number
INF = 1000000.0  # the largest value we consider a number

MAX_DEPTH = 4  # max ray bounces
RENDER_WIDTH = 64
RENDER_HEIGHT = 64
SPP = 5  # sample per pixel

scene = []


def normalize(vector):
    """
    returns a normalized unit vector in the same direction. For some reason numpy did not have this already.
    :param vector: the given vector
    :return: a normalized unit vector in the same direction
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

    def intersect(self, ray):
        """
        Tests for intersection between object and ray.
        https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
        :param ray:
        :return: the distance and the normal of the exact hit point.
        """
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

    def intersect(self, ray):
        """
        Tests for intersection between object and ray.
        https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
        :param ray:
        :return: the distance and the normal of the exact hit point
        """
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

    def orient_ray(self, x, y):
        """
        we determine the initial ray direction by simply adding a bit of left-ness and a bit of top-ness
        to our camera's forward direction (depending on the pixel coordinates)
        :param x: ray direction x
        :param y: ray direction y
        :return: re-normalizing to get a direction out of the ray
        """
        view_space_x = (x - (RENDER_WIDTH / 2.0)) * (self.view_plane_width / RENDER_WIDTH)
        view_space_y = (y - (RENDER_HEIGHT / 2.0)) * (self.view_plane_height / RENDER_HEIGHT)
        direction = (self.u * view_space_x) + (self.v * view_space_y) + (self.w * self.view_plane_dist)
        return normalize(direction)

    def simulate(self):
        """
        Gather "Samples Per Pixel" amount of samples for each pixel
        :return: pixel samples
        """
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


def hemisphere_dir(u1, u2):
    """
    helper function to go from two x,y uniform samples to a polar coordinate on the hemisphere.
    :param u1: x direction
    :param u2: y direction
    :return: the vector hit on the hemisphere
    """
    z = pow(1.0 - u1, 1.0)
    phi = 2 * np.pi * u2
    theta = np.sqrt(max(0.0, 1.0 - z * z))

    p = np.array([theta * np.cos(phi), theta * np.sin(phi), z])
    return p


def orient_hemisphere(p, normal):
    """
    convert a random hemisphere sample to a world-space ray direction
    :param p: input vector
    :param normal: normal vector
    :return: normalized ray vector
    """
    # create orthonormal basis around normal
    w = normal
    if abs(w[0]) > 0.1:
        u = np.array([0.0, 1.0, 0.0])
    else:
        u = np.array([1.0, 0.0, 0.0])
    u = np.cross(u, w)
    u = normalize(u)
    v = np.cross(w, u)

    # express sample in new coordinate basis
    ray_dir = (u * p[0]) + (v * p[1]) + (w * p[2])
    return normalize(ray_dir)  # normalized


def trace_path(ray, depth):
    """
    Recursive ray walk function
    :param ray: given ray
    :param depth: the number of ray bounces
    :return: the color represented in rgb
    """
    color = np.array([0.0, 0.0, 0.0])  # initialize to remove warning
    normal = None  # initialize to remove warning
    hit_point = None  # initialize to remove warning
    hit_distance = INF
    index = -1

    # if we've gone on for far too long, just terminate.
    if depth > MAX_DEPTH:
        return color

    # find where our ray first "hits" by getting the nearest intersection point over all objects in the scene
    for i in range(0, len(scene)):
        hit_data = scene[i].intersect(ray)
        if hit_data[0] > 0.0:
            if hit_data[0] < hit_distance:
                hit_distance = hit_data[0]
                hit_point = ray.get_hit_point(hit_distance)
                normal = hit_data[1]
                index = i

    if index == -1:
        return color

    else:
        u1 = np.random.random_sample()  # Generate two random samples to feed the hemisphere random sample generator
        u2 = np.random.random_sample()
        sample = hemisphere_dir(u1, u2)  # Generate hemisphere sample
        ray_dir = orient_hemisphere(sample, normal)  # Convert to world-space direction using our hit object

        incoming_ray = Ray(hit_point, ray_dir)  # Construct new ray from generated direction

        # We weight with probability p to terminate after the initial reflectance ray (depth 2+).
        kill_probability = 0.5
        if depth > 2 and np.random.random_sample() < kill_probability:
            return color

        light_contribution = trace_path(incoming_ray, depth + 1) # Recursive single sample of incoming light
        object_color = scene[index].diff
        object_emit = scene[index].emit
        brdf = object_color / np.pi  # Lambert BRDF

        # Apply rendering equation using the sample above
        color = 2 * np.pi * brdf * light_contribution * np.dot(ray_dir, normal) / kill_probability + object_emit

    return color  # return final single-sample color estimation

# =================================================
# MAIN FUNCTION
# Describe a scene for Monte Carlo Simulation to do its work


BLACK = np.array([0.0, 0.0, 0.0])
WHITE = np.array([1.0, 1.0, 1.0])
RED = np.array([1.0, 0.0, 0.0])
GREEN = np.array([0.0, 1.0, 0.0])
BLUE_EMIT = np.array([0.0, 0.0, 2.0])
WHITE_EMIT = np.array([1.2, 1.2, 1.2])

# Ground
plane_ground = Plane(np.array([0.0, -32.0, 0.0]), np.array([0.0, 1.0, 0.0]), WHITE, BLACK)
scene.append(plane_ground)

# Ceiling
plane_ceil = Plane(np.array([0.0, 32.0, 0.0]), np.array([0.0, -1.0, 0.0]), BLACK, WHITE_EMIT)
scene.append(plane_ceil)

# Left Wall
plane_left = Plane(np.array([-32.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), RED, BLACK)
scene.append(plane_left)

# Right Wall
plane_right = Plane(np.array([32.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), GREEN, BLACK)
scene.append(plane_right)

# Front wall
plane_front = Plane(np.array([0.0, 0.0, -32.0]), np.array([0.0, 0.0, 1.0]), WHITE, BLACK)
scene.append(plane_front)

# Back Wall (out of view)
plane_back = Plane(np.array([0.0, 0.0, 32.0]), np.array([0.0, 0.0, -1.0]), WHITE, BLACK)
scene.append(plane_back)

# Spheres
sphere_1 = Sphere(np.array([-18.0, -16.0, 16.0]), 16.0, WHITE, BLACK)
scene.append(sphere_1)

sphere_2 = Sphere(np.array([0.0, -30.0, 8.0]), 4.0, BLACK, BLUE_EMIT)
scene.append(sphere_2)

# Create camera and run the simulation
camera_origin = np.array([0.0, 0.0, -32.0])
vp_dist = 90
view_width = 200
view_height = 200
cam = Camera(camera_origin, vp_dist, view_width, view_height)
mc_pixel_samples = cam.simulate()


expected_pixels = np.empty((RENDER_WIDTH, RENDER_HEIGHT, 3))
for x in range(0, RENDER_WIDTH):
    for y in range(0, RENDER_HEIGHT):
        mean_pixel = np.array([0.0, 0.0, 0.0])
        for s in range(0, SPP):
            mean_pixel = mean_pixel + mc_pixel_samples[x, y, s,]
        mean_pixel = mean_pixel / SPP
        expected_pixels[x, y, ] = mean_pixel

# And now, if we pass it off to the image viewer, we can get an image of predicted (simulated) colors.
plt.imshow(np.rot90(expected_pixels), interpolation='gaussian')
plt.show(block=True)

# pixel_samples is stored like this: pixel_samples[x, y, s, c], where
# x and y are the pixel coordinates
# s is the s'th sample for that pixel
# and c is either 0, 1, or 2, which corresponds to the red, green, and blue components of the single sample.

