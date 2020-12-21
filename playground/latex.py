import math
import dill
import latexify
import ast
import inspect


def lat_name(n):
    return r"\mathsf{%s}" % n


def lat_nbin(n, s):
    return r"\mathbin{\mathop{%s}\limits_{%s}}" % (s, lat_name(n))


def lat_nfun(n, f):
    return r"\mathop{\mathop{\mathrm{%s}}\limits_{%s}}" % (f, lat_name(n))


FUNCTIONS = {
    "ndot": ("bin", lambda ax: lat_nbin(ax, r"\boldsymbol\cdot")),
    "sum": ("contraction", lambda ax: r"\sum\limits_{%s}" % lat_name(ax)),
    #    "softmax" : ("function", lambda ax: lat_nfun(ax, "softmax"))
}

MATH_SYMBOLS = {
    "aleph",
    "alpha",
    "beta",
    "beth",
    "chi",
    "daleth",
    "delta",
    "digamma",
    "epsilon",
    "eta",
    "gamma",
    "gimel",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "omega",
    "omega",
    "phi",
    "pi",
    "psi",
    "rho",
    "sigma",
    "tau",
    "theta",
    "upsilon",
    "varepsilon",
    "varkappa",
    "varphi",
    "varpi",
    "varrho",
    "varsigma",
    "vartheta",
    "xi",
    "zeta",
    "Delta",
    "Gamma",
    "Lambda",
    "Omega",
    "Phi",
    "Pi",
    "Sigma",
    "Theta",
    "Upsilon",
    "Xi",
}

BUILTIN_CALLEES = {
    "abs": (r"\left|{", r"}\right|"),
    "math.acos": (r"\arccos{\left({", r"}\right)}"),
    "math.acosh": (r"\mathrm{arccosh}{\left({", r"}\right)}"),
    "math.asin": (r"\arcsin{\left({", r"}\right)}"),
    "math.asinh": (r"\mathrm{arcsinh}{\left({", r"}\right)}"),
    "math.atan": (r"\arctan{\left({", r"}\right)}"),
    "math.atanh": (r"\mathrm{arctanh}{\left({", r"}\right)}"),
    "math.ceil": (r"\left\lceil{", r"}\right\rceil"),
    "math.cos": (r"\cos{\left({", r"}\right)}"),
    "math.cosh": (r"\cosh{\left({", r"}\right)}"),
    "math.exp": (r"\exp{\left({", r"}\right)}"),
    "math.fabs": (r"\left|{", r"}\right|"),
    "math.factorial": (r"\left({", r"}\right)!"),
    "math.floor": (r"\left\lfloor{", r"}\right\rfloor"),
    "math.fsum": (r"\sum\left({", r"}\right)"),
    "math.gamma": (r"\Gamma\left({", r"}\right)"),
    "math.log": (r"\log{\left({", r"}\right)}"),
    "math.log10": (r"\log_{10}{\left({", r"}\right)}"),
    "math.log2": (r"\log_{2}{\left({", r"}\right)}"),
    "math.prod": (r"\prod \left({", r"}\right)"),
    "math.sin": (r"\sin{\left({", r"}\right)}"),
    "math.sinh": (r"\sinh{\left({", r"}\right)}"),
    "math.sqrt": (r"\sqrt{", "}"),
    "math.tan": (r"\tan{\left({", r"}\right)}"),
    "math.tanh": (r"\tanh{\left({", r"}\right)}"),
    "sum": (r"\sum \left({", r"}\right)"),
}


BIN_OP_PRIORITY = {
    ast.Add: 10,
    ast.Sub: 10,
    ast.Mult: 20,
    ast.MatMult: 20,
    ast.Div: 20,
    ast.FloorDiv: 20,
    ast.Mod: 20,
    ast.Pow: 30,
}


def tname(n):
    if "," in n:
        return n[6:-8]
    return n


class NamedVisitor(latexify.core.LatexifyVisitor):
    def visit_FunctionDef(self, node):
        name_str = r"\mathrm{" + str(node.name) + "}"
        arg_strs = [self._parse_math_symbols(str(arg.arg)) for arg in node.args.args]

        def ann_str(ann):
            # t = r" \times ".join([self.visit(ann)  for ann in extra.annotation.elts])
            t = r" \times ".join(
                [lat_name(str(ann.id)) for ann in extra.annotation.elts]
            )
            return r"\mathbb{R}^{" + t + r"}"

        type_str = ""
        ts = []
        for name, extra in zip(arg_strs, node.args.args):
            if extra.annotation is not None:
                ts.append(ann_str(extra.annotation.elts))
        type_str = (
            name_str
            + ":"
            + r" \times ".join(ts)
            + r" \rightarrow "
            + ann_str(node.returns)
            + r"\\ "
            + "\n"
        )
        body_str = self.visit(node.body[0])
        return type_str + name_str + "(" + ", ".join(arg_strs) + r") = " + body_str

    def visit_Call(self, node):  # pylint: disable=invalid-name
        """Visit a call node."""
        callee_str = self.visit(node.func)
        arg_strs = [self.visit(arg) for arg in node.args]
        if callee_str in FUNCTIONS:
            a, func = FUNCTIONS[callee_str]
            if a == "bin":
                return arg_strs[1] + func(tname(arg_strs[0])) + arg_strs[2]
            if a == "contraction":
                return func(tname(arg_strs[0])) + arg_strs[1]
            if a == "function":
                return (
                    func(tname(arg_strs[0]))
                    + r"\left( "
                    + ", ".join(arg_strs[1:])
                    + r"\right)"
                )
        lstr, rstr = BUILTIN_CALLEES.get(callee_str, (None, None))
        if lstr is None:
            if callee_str.startswith("math."):
                callee_str = callee_str[5:]
            lstr = r"\mathrm{" + callee_str + r"}\left("
            rstr = r"\right)"
        return lstr + ", ".join(arg_strs) + rstr


def get_latex(fn, math_symbol=True):
    if hasattr(fn, "__latex_source__"):
        return fn.__latex_source__
    try:
        source = inspect.getsource(fn)
    # pylint: disable=broad-except
    except Exception:
        # Maybe running on console.
        source = dill.source.getsource(fn)
    return NamedVisitor(math_symbol=math_symbol).visit(ast.parse(source))
