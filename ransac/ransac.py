# coding=utf-8
# file: ransac.py
from data_struct import *

class Ransac(object):
    """
    Wrap the overall process, perform a role as a caller / manager
    """
    def __init__(self):
        pass


class PlaneEstimator(object):
    """
    Given 3 points, fit an exact plane
    Given 3+ points, use PCA to fit a plane
    """
    def __init__(self):
        pass
    
    @staticmethod
    def fit_exact_plane(points):
        """
        This function assumes the points could actually fit a plane, and throw an error if not
        :param points: points used to fit a plane
        :return plane: exactly fitted plane
        """
        assert len(points) > 2, len(points)
        p0, p1, p2 = points[:3]
        vec_01 = (p0 - p1).coordinate
        vec_02 = (p0 - p2).coordinate
        plane_norm = np.cross(vec_01, vec_02)
        offset = np.dot(plane_norm, p0.coordinate)
        plane = Plane.from_norm_offset(plane_norm, offset)
        # check whether the rest points are on the plane
        flag = True
        for p in points[3:]:
            if p not in plane:
                flag = False
                break
        if not flag:
            raise ValueError('given points cannot fit an exact plane')
        return plane

    @staticmethod
    def fit_approx_plane(points):
        """
        This function estimate a plane via PCA
        :param points: points used to fit a plane
        :return plane: approximately fitted plane
        """

if __name__ == "__main__":
    points = [
        [-4, 0, 0],
        [0, -2, 0],
        [-2, -1, 0],
    ]
    points = [Point.from_coordinate(point) for point in points]
    _ = [print(point) for point in points]
    print(points[0])
    print(Point(1, 2, 3))
    print(dir(Point(1, 2, 3)))
    print(dir(points[0]))
    pass