from IPython.display import Math

ndot = lambda _: None
nfunsoftmax = lambda _: None


def Att(Q: (key,), K: (seq, key), V: (val,)) -> (val,):
    return ndot(seq, softmax(seq, ndot(key, Q, K) / math.sqrt(abs(key))), V)


Math(get_latex(Att))

"""# Extend functions to work with tensors



`tensorize` and `scalar`
"""


def scalar(f):

    """Make a scalar R-->R function into a tensor one"""

    def g(X: tuple()):

        return Scalar(f(Scalar(X)))

    return g


import inspect


def tensorize(f):

    """Takes function f with type annotation of the form X:(ax1,ax,)....

      Transforms it to function that can handle tensors with dangling and aligned axes

      This is a somewhat brittle implementation - should fix this"""

    S = inspect.signature(f)

    template = {}

    for arg in S.parameters:

        if S.parameters[arg].annotation and isinstance(
            S.parameters[arg].annotation, (tuple, list)
        ):

            template[arg] = S.parameters[arg].annotation

    def g(*L, **kwd):

        ba = S.bind(*L, **kwd)

        spare_axes = collections.defaultdict(list)

        found = False

        slices = []

        for (arg, axes) in template.items():

            if set(ba.arguments[arg].axes) != set(axes):

                if set(axes).issubset(set(ba.arguments[arg].axes)):

                    for ax in set(ba.arguments[arg].axes):

                        if not ax in set(axes):

                            spare_axes[arg].append(ax)

                            slices.append(ba.arguments[arg].slice(ax))

                            found = True

            if not found:

                return f(*ba.args, **ba.kwargs)

            D = {}

            shape = slices

            L = [[s.type(i) for i in s.range] for s in slices]

            orig_arguments = {arg: ba.arguments[arg].copy() for arg in ba.arguments}

            for k in itertools.product(*L):

                AD = dict(orig_arguments)

                for arg in AD:

                    if arg in spare_axes:

                        tempk = tuple([a for a in k if type(a) in spare_axes[arg]])

                        AD[arg] = AD[arg][tempk]

                ba.arguments.update(AD)

                Y = f(*ba.args, **ba.kwargs)

                if isinstance(Y, (float, int)):

                    temp = tuple(sorted(k))

                    D[temp] = Y

                else:

                    for k_, v in Y.items():

                        D[tuple(sorted(list(k) + list(k_)))] = v

                    for s in D[tuple(sorted(k))].shape:

                        if not s in shape:

                            shape.append(s)

            return Tensor(shape, D)

    g.__signature__ = S

    try:

        g.__latex_source__ = get_latex(f)

    except Exception:

        g.__latex_source__ = "Can't extract"

    return g


def sum(ax, A):
    def f(X: (ax,)):

        res = 0

        for v in X.values():

            res += v

        return res

    return tensorize(f)(A)


sum(foo, A)


@scalar
def exp(x):

    return math.exp(x)


def softmax(ax, A):
    def f(X: (ax,)):

        return exp(X) / sum(ax, X)

    return tensorize(f)(A)


softmax(foo, A)


def sum(ax, A):
    def f(X: (ax,)):

        res = 0

        for v in X.values():

            res += v

        return res

    return tensorize(f)(A)


sum(foo, A)
