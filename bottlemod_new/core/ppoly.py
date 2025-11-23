from __future__ import annotations

import bisect
import heapq
import itertools
import math
from typing import Any, List, Optional, Union

import numpy
import scipy.interpolate


# Class representing a piecewise polynomial function.
# Based on PPoly from SciPy.
class PPoly(scipy.interpolate.PPoly):
    """Initialize PPoly with breakpoints at x and function coefficients c.

    Parameters
    ----------
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.

    Raises
    ------
    ValueError
        If `x` is not strictly increasing.
    """

    def __init__(self, x: numpy.ndarray, c: numpy.ndarray) -> None:
        x = numpy.array(x, dtype="object").flatten()
        if (len(x) > 1) and (x[0] > x[1]):
            raise ValueError("`x` must be strictly increasing.")
        while len(c) > 1 and all([cc == 0 for cc in c[0]]):
            c = c[1:]
        super().__init__(c, x)

    @staticmethod
    def __addPoly(c1: numpy.ndarray, c2: numpy.ndarray) -> numpy.ndarray:
        if len(c1) == len(c2):
            return c1 + c2
        elif len(c1) < len(c2):
            return PPoly.__addPoly(c2, c1)
        else:  # len(c1) > len(c2)
            return c1 + numpy.append(numpy.zeros((len(c1) - len(c2), 1)), c2, axis=0)

    @staticmethod
    def __mulPoly(c1: numpy.ndarray, c2: numpy.ndarray) -> numpy.ndarray:
        newc = numpy.zeros((len(c1) + len(c2) - 1, 1))
        for i, v1 in enumerate(reversed(c1)):
            for j, v2 in enumerate(reversed(c2)):
                newc[len(newc) - 1 - (i + j)][0] += v1[0] * v2[0]
        return newc

    @staticmethod
    def __insertPoly(x: float, outer: "PPoly", otherpows: list["PPoly"]) -> list[Any]:
        result: list[Any] = []
        for i, coef in enumerate(reversed(outer[otherpows[1](x)].c)):
            relevantPow = otherpows[i][x]
            # result = numpy.array(list(itertools.chain(itertools.repeat([0], len(relevantPow.c) + i - len(result)), result))) + numpy.array(list(itertools.chain(coef * relevantPow.c, itertools.repeat([0], i))))
            result = (
                numpy.array(
                    list(
                        itertools.chain(
                            itertools.repeat([0], len(relevantPow.c) - len(result)),
                            result,
                        )
                    )
                )
                + coef * relevantPow.c
            )
        return result

    @staticmethod
    def __insertPolyOld(x: Any, outer: Any, otherpows: Any) -> list[Any]:
        result: list[Any] = []
        for i, coef in enumerate(reversed(outer[x].c)):
            relevantPow = otherpows[i][x]
            # result = numpy.array(list(itertools.chain(itertools.repeat([0], len(relevantPow.c) + i - len(result)), result))) + numpy.array(list(itertools.chain(coef * relevantPow.c, itertools.repeat([0], i))))
            result = (
                numpy.array(
                    list(
                        itertools.chain(
                            itertools.repeat([0], len(relevantPow.c) - len(result)),
                            result,
                        )
                    )
                )
                + coef * relevantPow.c
            )
        return result

    # Unary minus operator.
    def __neg__(self) -> "PPoly":
        return PPoly(self.x, -self.c)

    # Binary plus operator.
    # Calculating the sum of two functions.
    def __add__(self, other: Union["PPoly", Any]) -> "PPoly":
        if isinstance(other, PPoly):
            newx = []
            newc = [[]] * max(len(self.c), len(other.c))
            lastx = None
            lastpoly = []
            for x in list(heapq.merge(self.x, other.x))[0:-1]:
                if x != lastx:
                    newpoly = PPoly.__addPoly(self[x].c, other[x].c)
                    if len(newpoly) != len(lastpoly) or any(newpoly != lastpoly):
                        lastpoly = newpoly
                        newx.append(x)
                        newc = [
                            c1 + c2
                            for (c1, c2) in zip(
                                newc,
                                [[0]] * (len(newc) - len(newpoly)) + newpoly.tolist(),
                            )
                        ]
                        lastx = x
            return PPoly(newx + [max(self.x[-1], other.x[-1])], newc)
        else:
            return self + PPoly([self.x[0], self.x[-1]], numpy.array(other, ndmin=2))

    # Binary minus operator.
    # Subtracting a function from another.
    def __sub__(self, other: Union["PPoly", Any]) -> "PPoly":
        return self + -other

    # Multiplication operator.
    # Multiplies two functions or a function with a scalar.
    def __mul__(self, other: Union["PPoly", Any]) -> "PPoly":
        if isinstance(other, PPoly):
            newx: List[Any] = []
            newc: List[Any] = [[]] * (len(self.c) + len(other.c))
            lastx: Optional[Any] = None
            lastpoly: List[Any] = []
            for x in list(heapq.merge(self.x, other.x))[0:-1]:
                if x != lastx:
                    newpoly = PPoly.__mulPoly(self[x].c, other[x].c)
                    if len(newpoly) != len(lastpoly) or any(newpoly != lastpoly):
                        lastpoly = newpoly
                        newx.append(x)
                        newc = [
                            c1 + c2
                            for (c1, c2) in zip(
                                newc,
                                [[0]] * (len(newc) - len(newpoly)) + newpoly.tolist(),
                            )
                        ]
                        lastx = x
            return PPoly(newx + [max(self.x[-1], other.x[-1])], newc)
        else:
            return PPoly(self.x, self.c * other)

    def __rmul__(self, other: Any) -> "PPoly":
        return self * other

    # Find the common range where functions x1 and x2 are both defined.
    @staticmethod
    def get_intersection_range(x1: Any, x2: Any) -> numpy.ndarray:
        minx = max(x1[0], x2[0])
        maxx = min(x1[-1], x2[-1])
        return numpy.unique(
            [x for x in itertools.chain(x1, x2) if x >= minx and x <= maxx]
        )

    # Division operator.
    # Divide a function by another, piecewise constant function or a scalar.
    def __truediv__(self, other: Union["PPoly", Any]) -> "PPoly":
        if isinstance(other, PPoly):
            if other.c.shape[0] != 1:
                raise ValueError(
                    "Divisor of PPoly division must be a piecewise constant function."
                )
            newx = PPoly.get_intersection_range(self.x, other.x)
            newc: List[Any] = [[]] * len(self.c)
            lastpoly: List[Any] = []
            for x in newx[0:-1]:
                newpoly = self[x].c / other[x].c[0][0]
                if len(newpoly) != len(lastpoly) or any(newpoly != lastpoly):
                    lastpoly = newpoly
                    newc = [
                        c1 + c2
                        for (c1, c2) in zip(
                            newc, [[0]] * (len(newc) - len(newpoly)) + newpoly.tolist()
                        )
                    ]
            return PPoly(newx, newc)
        else:
            return PPoly(self.x, self.c / other)

    # Allow conversion to a string, formatting the piecewise function into a readable format.
    def __str__(self) -> str:
        result = ""
        x = ["{:.2f}".format(val) for val in self.x]
        maxlen = max([len(val) for val in x])
        for i, val in enumerate(self.c.T):
            result += " " * (maxlen - len(x[i]))
            result += x[i]
            result += " - "
            result += " " * (maxlen - len(x[i + 1]))
            result += x[i + 1]
            result += ": "
            for i, c in enumerate(val):
                result += "{}".format(c)
                result += "*x^^("
                result += "{}".format(len(val) - 1 - i)
                result += ") + "
            result = result[:-3]
            result += "\n"
        return result

    # Call operator.
    # Can be called with two types of arguments:
    # 1. Another instance of PPoly: creates a new function by inserting other into this function; basically result = self(other(x)). Argument "nu" is ignored.
    # 2. A scalar: evaluates the function at the specific location. "nu" is the order of derivative to evaluate.
    def __call__(self, other: Any, nu: int = 0) -> Any:
        if isinstance(other, PPoly):
            otherpows: List["PPoly"] = [PPoly([-math.inf, math.inf], [[1]]), other]
            for _ in range(len(self.c) - 1):
                otherpows.append(otherpows[-1] * other)

            # get relevant segment change positions of outer
            outerchanges: List[Any] = []
            for x in self.x:
                changes = other.solve(
                    x
                )  # might only work if crossed increasingly -> only for monotonically increasing functions
                changes = [
                    c
                    for c in sorted(changes)
                    if not math.isnan(c) and abs(other(c) - x) < 0.0001
                ]  # bug in scipy regarding solve, please fix

                if len(changes) > 0:
                    outerchanges = outerchanges + [changes[-1]]

            newx: List[Any] = []
            newc: List[Any] = [[]] * (len(self.c) * len(other.c))  # todo: check bounds
            lastx: Optional[Any] = None
            lastpoly: List[Any] = []
            for x in list(heapq.merge(outerchanges, other.x))[0:-1]:
                if x != lastx:
                    newpoly = PPoly.__insertPoly(x, self, otherpows)
                    if len(newpoly) != len(lastpoly) or any(newpoly != lastpoly):
                        lastpoly = newpoly
                        newx.append(x)
                        newc = [
                            c1 + c2
                            for (c1, c2) in zip(
                                newc,
                                [[0]] * (len(newc) - len(newpoly)) + newpoly.tolist(),
                            )
                        ]
                        lastx = x
            return PPoly(newx + [max(outerchanges[-1], other.x[-1])], newc)
        else:
            return super().__call__(other, nu)

    # Find roots of the function; x values where f(x) is zero.
    # If discontinuity is True, those x values are treated as roots, where a new piece starts, the end of the previous piece evaluates > 0 and the start of the next piece evaluates < 0, or vice versa.
    def roots(self, discontinuity: bool = True) -> List[Any]:
        result = list(super().roots(False))
        if discontinuity:
            lastsector = self[self.x[0]]
            for bp in self.x[1:-1]:
                thissector = self[bp]
                before = lastsector(bp)
                after = thissector(bp)
                if (before < 0 and after > 0) or (before > 0 and after < 0):
                    result.append(bp)
                lastsector = thissector
        return result

    # Index read operator. Returns part of the function.
    # If indexed by a scalar x: return the piece belonging to x as a PPoly.
    # If indexed by a range / slice x: returns the function inside the range (one or more pieces).
    def __getitem__(self, x: Union[slice, Any]) -> "PPoly":
        if isinstance(x, slice):
            if x.step is not None:
                raise ValueError("Slices with steps not supported.")
            startpos = x.start if x.start is not None else self.x[0]
            endpos = x.stop if x.stop is not None else self.x[-1]
            startidx = bisect.bisect(self.x, startpos)
            startidx = startidx - 1 if startidx > 0 else 0
            endidx = bisect.bisect(self.x, endpos)
            endidx = endidx - 1 if endidx > 0 else 0
            newx = list(self.x[startidx : endidx + 1])
            newx[0] = startpos
            if endpos > self.x[-1] and len(newx) > 1:
                newx[-1] = endpos
            elif endpos not in self.x:
                newx.append(endpos)
            startidx = startidx if startidx < len(self.c[0]) else len(self.c[0]) - 1
            endidx = endidx if endidx < len(self.c[0]) else len(self.c[0]) - 1
            endidx = endidx - 1 if endpos in self.x[:-1] else endidx
            return PPoly(newx, [e[startidx : endidx + 1] for e in self.c])
        else:
            idx = bisect.bisect(self.x, x)
            idx = idx - 1 if idx > 0 else 0
            startpos = (
                self.x[len(self.x) - 2]
                if idx >= len(self.x) - 1
                else self.x[idx]
                if idx > 0
                else -math.inf
            )
            endpos = self.x[idx + 1] if idx + 1 < len(self.x) else math.inf
            return PPoly(
                [startpos, endpos],
                [[val[idx if idx < len(val) else len(val) - 1]] for val in self.c],
            )

    # (Internal helper function.)
    # Compares two sets of function coefficients. True if they describe the same function.
    @staticmethod
    def _equal_coefficients(a: numpy.ndarray, b: numpy.ndarray) -> bool:
        if len(a) == len(b):
            return all(a == b)
        elif len(a) < len(b):
            a, b = b, a  # a is always longer than b
        return all(a[0 : len(a) - len(b)] == numpy.zeros(len(a) - len(b))) and all(
            a[len(a) - len(b) :] == b
        )

    # Index write operator.
    # Overwrites part of the function defined by the range / slice x with the function val (at the same range).
    def __setitem__(self, x: slice, val: "PPoly") -> None:
        startpos = x.start if x.start is not None else -math.inf
        endpos = x.stop if x.stop is not None else math.inf

        if startpos > self.x[0]:
            result = self[:startpos]
            insert_ppoly = val[startpos:endpos]
            if not self._equal_coefficients(result.c.T[-1], insert_ppoly.c.T[0]):
                result.extend(insert_ppoly.c, insert_ppoly.x[1:])
            else:
                result.x[-1] = insert_ppoly.x[1]
                if len(insert_ppoly.x) > 2:
                    insert_ppoly = PPoly(insert_ppoly.x[1:], insert_ppoly.c.T[1:].T)
                    result.extend(insert_ppoly.c, insert_ppoly.x[1:])
        else:
            result = val[startpos:endpos]

        if self.x[-1] > endpos:
            insert_ppoly = self[endpos:]
            if not self._equal_coefficients(result.c.T[-1], insert_ppoly.c.T[0]):
                result.extend(insert_ppoly.c, insert_ppoly.x[1:])
            else:
                result.x[-1] = insert_ppoly.x[1]
                if len(insert_ppoly.x) > 2:
                    insert_ppoly = PPoly(insert_ppoly.x[1:], insert_ppoly.c.T[1:].T)
                    result.extend(insert_ppoly.c, insert_ppoly.x[1:])

        self.x = result.x
        self.c = result.c

    def get_next_change(self, x: float) -> float:
        pos = bisect.bisect(self.x, x)  # numpy.searchsorted
        pos = pos if pos != 0 and pos != len(self.x) - 1 else pos + 1
        return self.x[pos] if pos < len(self.x) else math.inf
