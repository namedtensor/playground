import functools

AXES = []


@functools.total_ordering
class Meta(type):
    def __getitem__(self, arg):
        return Slice(self, arg)

    def __lt__(self, other):
        return str(self) < str(other)


@functools.total_ordering
class Slice:
    def __init__(self, cls, arg):
        self.range = [cls(i) for i in range(arg)]
        self.type = cls

    def __lt__(self, other):
        return (str(self.type), len(self.range)) < (str(other.type), len(other.range))

    def name(self):
        t = str(self.type)
        return t[t.rfind(".") + 1 : t.rfind("'")]

    def __str__(self):
        return self.name() + f"[{len(self.range)}]"

    def __repr__(self):
        return str(self)


@functools.total_ordering
class Index(int, metaclass=Meta):
    def __new__(cls, value):
        x = int.__new__(cls, value)
        return x

    def __lt__(self, other):
        return (str(type(self)), int(self)) < (str(type(other)), int(other))

    def __eq__(self, other):
        return str(type(self)), int(self) == str(type(other)), int(other)

    def __hash__(self):
        return hash(str(type(self)) + str(int(self)))

    def __str__(self):
        return self.name() + f"({int(self)})"

    def __repr__(self):
        return str(self)

    def name(self):
        t = str(type(self))
        return t[t.rfind(".") + 1 : t.rfind("'")]


def ClassFactory(name, BaseClass=Index):
    def __init__(self, ind):
        Index.__init__(self)

    newclass = type(name, (Index,), {"__init__": __init__})
    return newclass


def setup_axes(*axes):
    global AXES
    classes = []
    for ax in axes:
        if ax in AXES:
            continue
        AXES.append(ax)
        classes.append(ClassFactory(ax))
    return classes
