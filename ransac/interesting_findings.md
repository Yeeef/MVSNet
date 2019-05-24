# interesting findings

```python

class Test1(object):
    def __init__(self, a, b):
        self.a, self.b = a, b
    @classmethod
    def from_tuple(cls, t):
        # difference is here
        cls.a, cls.b = t
        return cls
    def __str__(self):
        return 'test: (%f, %f)' % (self.a, self.b)

test1 = Test1(1, 2)
test2 = Test1.from_tuple((1, 2))
print(test1, test2)
```