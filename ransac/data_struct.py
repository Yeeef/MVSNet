# coding=utf-8
# File: data_struct.py
"""
Declare and define necessary data struct
"""

import numpy as np

class Vector(object):
    """
    Class wrap all vector functionality, basically it just calls numpy package
    """
    def __init__(self, coordinate):
        assert isinstance(coordinate, (np.ndarray, list, tuple)), type(coordinate)
        assert len(coordinate) == 3, len(coordinate)
        self.coordinate = np.array(coordinate)
    
    def __add__(self, other):
        return Vector(self.coordinate + other.coordinate)
    
    def __sub__(self, other):
        return Vector(self.coordinate + other.coordinate)
    
    def __iter__(self):
        for ax in self.coordinate:
            yield ax

    def __len__(self):
        return 3


class Point(object):
    def __init__(self, x, y, z):
        """
        3D point
        """
        self.x = x
        self.y = y
        self.z = z
        self.coordinate = np.array([x, y, z])

    @classmethod
    def from_coordinate(cls, coordinate):
        assert len(coordinate) == 3, len(coordinate)
        return cls(*coordinate)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __iter__(self):
        for ax in self.coordinate:
            yield ax

    def __str__(self):
        return 'Point: (%f, %f, %f)' % (self.x, self.y, self.z)


class Plane(object):
    def __init__(self, a, b, c, d):
        """
        ax + by + cz + d = 0
        """
        self.a, self.b, self.c, self.d = a, b, c, d

    def norm(self):
        return Vector([self.a, self.b, self.c])

    def offset(self):
        return self.d

    @classmethod
    def from_norm_offset(cls, norm_vec, offset):
        cls.a, cls.b, cls.c = norm_vec
        cls.d = offset
        return cls
    
    def __contains__(self, point):
        assert isinstance(point, (np.ndarray, Point)), type(point)
        x, y, z = point
        return not np.dot(
            np.array([x, y, z, 1]),
            np.array([self.a, self.b, self.c, self.d])
        )

    def __str__(self):
        return 'Plane: %f * x + %f * y + %f * z + %f = 0' % (self.a, self.b, self.c, self.d)

if __name__ == "__main__":
    plane = Plane(1, 2, 3, 4)
    point = Point(-4, 0, 0)
    p2 = Point(0, 0, 0)
    p3 = np.array([-4, 0, 0])
    print(plane)
    print(point in plane, p2 in plane, p3 in plane)
    print(point)
