from bottlemod.ppoly import PPoly


# Representing a piecewise polynomial function that is monotonically increasing.
# Simply adds sanity checks for the monotonical increase to PPoly.
class Func(PPoly):
    def __init__(self, x: list, c: list):
        super().__init__(x, list(map(list, zip(*c))))

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
