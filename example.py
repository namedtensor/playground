from playground import *

foo, bar, baz, key, seq, val, ax = setup_axes(
    "foo", "bar", "baz", "key", "seq", "val", "ax"
)

A = Tensor([foo[4], bar[5]])
for i in A.range(foo):
    for j in A.range(bar):
        A[i, j] = (i + 1) * (j + 1)
print(A)
B = Scalar(5)
print(B[foo(2), bar(3)])
print(3 + B)
print(A + B)
print(A[foo(2)])
