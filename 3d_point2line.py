from collections import namedtuple
Point = namedtuple('Point', ['x', 'y', 'z'])
Line = namedtuple('Line', ['a', 'b', 'c', 'd'])

def sample_point_on_line(line):
    *_, c, d = line.a, line.b, line.c, line.d
    assert c != 0, c
    x = y = 0
    z = -d / c
    return Point(x, y, z)


if __name__ == "__main__":
    point = Point(2, 3, 4)
    line = Line(1, 1, 1, 1)
    sample_point = sample_point_on_line(line)
    # pro