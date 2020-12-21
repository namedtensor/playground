import collections
import itertools
import types


def union(*L):
    shapes = []
    for shape in L:
        for s in shape:
            found = False
            for i, t in enumerate(shapes):
                if t.type == s.type:
                    if len(s.range) > len(t.range):
                        shapes[i] = s
                    found = True
                    break
            if not found:
                shapes.append(s)
    return tuple(sorted(shapes))


def keys(shape):
    L = [[s.type(i) for i in s.range] for s in shape]
    return itertools.product(*L)


class Tensor:
    def __init__(self, shape, D={}):
        if not shape:
            self.axes = tuple()
            self.shape = tuple()
            self.data = collections.defaultdict(int)
            return
        self.axes = tuple(sorted([s.type for s in shape]))
        self.shape = tuple(sorted(shape))
        self.data = collections.defaultdict(int)
        for k, v in D.items():
            self.data[tuple(sorted(k))] = v

    def range(self, ax):
        return self.slice(ax).range

    def keys(self):
        L = [[s.type(i) for i in s.range] for s in self.shape]
        return itertools.product(*L)

    def items(self):
        return [(k, self.data[k]) for k in self.keys()]

    def values(self):
        return [self.data[k] for k in self.keys()]

    def allkeys(self, other):
        return keys(union(self.shape, other.shape))

    def __add__(self, other):
        if isinstance(other, (int, float)):
            res = self.copy()
            for k, v in self.items():
                res.data[k] = v + other
            return res
        res = self.copy()
        for k in self.allkeys(other):
            res[k] = self[k] + other[k]
        return res

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            res = self.copy()
            for k, v in self.items():
                res.data[k] = v - other
            return res
        res = self.copy()
        for k in self.allkeys(other):
            res[k] = self[k] - other[k]
        return res

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            res = self.copy()
            for k, v in self.items():
                res.data[k] = v * other
            return res
        res = self.copy()
        for k in self.allkeys(other):
            res[k] = self[k] * other[k]
        return res

    def __div__(self, other):
        if isinstance(other, (int, float)):
            res = self.copy()
            for k, v in self.items():
                res.data[k] = v / other
            return res
        res = self.copy()
        for k in self.allkeys(other):
            res[k] = self[k] / other[k]
        return res

    def copy(self):
        return Tensor(self.shape, self.data)

    def slice(self, ax):
        for s in self.shape:
            if s.type == ax:
                return s
        raise KeyError()

    def restrictkey(self, key):
        if not isinstance(key, (tuple, list)):
            key = tuple([key])
        nk = [k for k in key if type(k) in self.axes]
        return tuple(sorted(nk))

    def fullkey(self, key):
        if not isinstance(key, (tuple, list)):
            key = tuple([key])
        keytype = tuple(sorted([type(k) for k in key]))
        for t in keytype:
            if t not in self.axes:
                raise KeyError(f"Type {t} not in axes: {self.axes}")
        return set(keytype) == set(self.axes)

    def __setitem__(self, key, val):
        if not isinstance(key, tuple):
            key = tuple([key])
        if self.fullkey(key) and isinstance(val, (float, int)):
            self.data[tuple(sorted(key))] = val
            return
        if isinstance(val, Tensor):
            for k, v in val.items():
                self.data[tuple(sorted(list(key) + list(k)))] = v
            return
        raise KeyError(f"{key}, {val} incompatible with shape of tensor")

    def __getitem__(self, key):
        key = self.restrictkey(key)
        if len(key) == len(self.axes):
            return self.data[tuple(sorted(key))]
        myaxes = [type(i) for i in key]
        shape = [s for s in self.shape if not s.type in myaxes]
        L = [[s.type(i) for i in s.range] for s in shape]
        D = {}
        for k in itertools.product(*L):
            D[tuple(sorted(k))] = self.data[tuple(sorted(list(key) + list(k)))]
        return Tensor(shape, D)

    def _repr_latex_(self):
        if len(self.shape) == 0:
            return "$" + str(self.scalar) + r"\in\mathbb{R}$"
        s = r"\times".join([fr"\mathsf{{{s.name()}}}" for s in self.shape])
        mytype = "\mathbb{R}^{" + s + "}"
        if len(self.shape) == 2:
            ax1 = self.shape[0].type
            ax2 = self.shape[1].type
            res = (
                f"$$\\mathsf{{{self.shape[0].name()}}}"
                + "\\begin{array}{c}"
                + mytype
                + r"\\ "
                + r"\mathsf{"
                + self.shape[1].name()
                + "} "
                + r"\\ \begin{bmatrix}"
            )
            for i in self.shape[0].range:
                res += (
                    " & ".join([str(self[ax1(i), ax2(j)]) for j in self.shape[1].range])
                    + r"\\"
                )
            res = res[:-2] + r" \end{bmatrix}\end{array}$$"
            return res
        if len(self.shape) == 1:
            ax1 = self.shape[0].type
            res = (
                f"$$"
                + "\\begin{array}{c}"
                + mytype
                + r"\\ "
                + r"\mathsf{"
                + self.shape[0].name()
                + "} "
                + r"\\ \begin{bmatrix}"
            )
            res += " & ".join([str(self[ax1(i)]) for i in self.shape[0].range])
            res += r"\end{bmatrix}\end{array}$$"
            return res
        return "$$" + mytype + "$$"


class Scalar(Tensor, float):
    def __new__(cls, value):
        return super(Scalar, cls).__new__(cls, value)

    def __init__(self, scalar):
        if isinstance(scalar, Scalar):
            self.scalar = scalar.scalar
        else:
            self.scalar = scalar
        super().__init__(None)

    def copy(self):
        return Scalar(self.scalar)

    def __getitem__(self, key):
        return self.scalar
