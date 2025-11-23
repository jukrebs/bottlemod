from __future__ import annotations
from collections.abc import Sequence

from .ppoly import PPoly


class Func(PPoly):
    """Monotonically increasing piecewise polynomial function.

    Extends PPoly with validation that the function is weakly monotonically
    increasing (non-decreasing).

    Parameters
    ----------
    x : list or array-like
        Breakpoints for the piecewise polynomial. Must be sorted in
        increasing order. Shape (m+1,) for m intervals.
    c : list of lists or array-like
        Polynomial coefficients for each interval. Shape (m, k) where
        k is the polynomial order. Each row contains coefficients for
        one interval in descending order [a_k, a_{k-1}, ..., a_1, a_0].

        Note: Coefficients are transposed internally to match PPoly's
        expected format (k, m).

    Raises
    ------
    ArithmeticError
        If the function is not monotonically increasing (has negative
        derivative at breakpoints or negative second derivative at
        critical points).

    Notes
    -----
    The monotonicity check works by:
    1. Computing first derivative and checking it's non-negative at breakpoints
    2. Finding roots of first derivative (critical points)
    3. Checking second derivative is non-negative at critical points
    """

    def __init__(
        self, x: Sequence[int | float], c: Sequence[Sequence[int | float]]
    ) -> None:
        super().__init__(x, list(map(list, zip(*c))))  # type: ignore

        # check for weak monotonic increase
        d1 = self.derivative()
        d2 = d1.derivative()

        for bpx in self.x[:-1]:
            if d1(bpx) < 0:
                raise ArithmeticError(
                    "Piecewise defined polynomial must be monotonically increasing."
                )

        for xp in d1.roots():
            if d2(xp) < 0:
                raise ArithmeticError(
                    "Piecewise defined polynomial must be monotonically increasing."
                )


if __name__ == "__main__":
    f = Func([-1, 1], [[1, 0]])
    print(f(0), f(-100), f(100))  # 0 -1 1
    f = Func([-10, -5, 1], [[0, 1, -100], [-1, 0, 0]])
