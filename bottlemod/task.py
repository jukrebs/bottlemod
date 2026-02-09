import bisect
import itertools
import math
from typing import Any, List, Tuple, Union, Optional

import numpy as np

from bottlemod.func import Func
from bottlemod.ppoly import PPoly


# Represents a task (or subtask, step, however the granularity in which BottleMod is being used), abstracted from its execution.
# cpu_funcs and data_funcs are the resource and data requirement functions of the task respectively.
class Task:
    def __init__(self, cpu_funcs: List[PPoly], data_funcs: List[Func]):
        self.cpu_funcs = cpu_funcs
        self.data_funcs = data_funcs

        max_data = cpu_funcs[0].x[-1]
        for cpu_func in cpu_funcs:
            if cpu_func.x[-1] != max_data:
                raise ValueError(
                    "All CPU functions must have the same range on x axis."
                )
        for data_func in data_funcs:
            if data_func(data_func.x[-1]) != max_data:
                raise ValueError("All data funcs must have the same end output.")


# Represents an execution of a task.
# This brings together the task (and with that its requirement functions) and the input functions provided by the execution environment or previous tasks.
class TaskExecution:
    # The constructor takes the task (which includes its requirement functions) and input functions and calculates the output.
    # Output functions as described in the paper are not implemented here, but semantically it would just be `output = task.outputFunc(taskExecution._real_out)`.
    def __init__(
        self, task: Task, in_cpu_funcs: List[PPoly], in_data_funcs: List[Func]
    ) -> None:
        self._task = task
        self._in_cpu_funcs = in_cpu_funcs
        self._in_data_funcs = in_data_funcs
        # todo: additional sanity checks
        if len(in_data_funcs) != len(self._task.data_funcs):
            raise ValueError(
                "Must have one input data function for each of the tasks output data function."
            )
        if len(in_cpu_funcs) != len(self._task.cpu_funcs):
            raise ValueError(
                "Must have one input cpu function for each of the tasks output cpu function."
            )

        # todo: for actual release that should probably be commented in
        # for our cases it'll fail because of rounding errors
        # for (in_data, out_data) in zip(in_data_funcs, self._task.data_funcs):
        #    if in_data(in_data.x[-1]) != out_data.x[-1]:
        #        raise ValueError('Input data function must match corresponding output data function.')

        self._real_out = None
        self._bottlenecks = []
        self.calculate()

    # Internal function returning the last segment of a function.
    @staticmethod
    def last_segment(func, x):
        last_section_index = bisect.bisect_left(func.x, x) - 1
        return (
            func[func.x[last_section_index]]
            if last_section_index >= 0
            else PPoly([0, 1], [[0]])
        )

    # Internal function.
    # Calculates the current speed a certain resource (`in_function` being its input function, `out_function` its requirement function) allows the task to make progress at a specific point in time t.
    @staticmethod
    def get_speed(
        in_function: PPoly, out_function: PPoly, t: float, progress: Union[PPoly, float]
    ) -> Tuple[PPoly, float]:
        in_function = TaskExecution.next_segment_only(in_function, t)
        if type(progress) is PPoly:
            if out_function.c.shape[0] == 1:
                out_segment = out_function
            else:
                out_segment = out_function(TaskExecution.next_segment_only(progress, t))
            if all(cc == 0 for cc in out_segment.c.flatten()):
                return (
                    PPoly([t, in_function.x[-1]], [[1e300]]),
                    out_function.x[-1],
                )
            return (
                in_function / out_segment,
                out_function.x[-1],
            )
        else:  # only works if out_function is piecewise constant, todo: sanity check that
            relevant_out_function = out_function[progress]
            if all(cc == 0 for cc in relevant_out_function.c.flatten()):
                return PPoly([t, in_function.x[-1]], [[1e300]]), math.inf
            max_progress_validity = relevant_out_function.x[-1]
            relevant_out_function.x[0], relevant_out_function.x[1] = (
                -math.inf,
                math.inf,
            )  # extend this as the actual limitation comes from the underlying progress and is handled elsewhere; different units
            result = in_function / relevant_out_function
            # todo: sanity check if len(result.x) == 2 ?
            return result, max_progress_validity

    # Internal helper to trim out problems with floating point rounding errors.
    @staticmethod
    def precision_threshold(val: float):
        return 40 * val * 10 ** (-15)

    @staticmethod
    def normalize_solutions(solutions) -> List[float]:
        if isinstance(solutions, np.ndarray):
            candidates = solutions.tolist()
        elif isinstance(solutions, (list, tuple)):
            candidates = list(solutions)
        else:
            candidates = [solutions]
        result = []
        for candidate in candidates:
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                result.append(value)
        result.sort()
        return result

    @staticmethod
    def estimate_time_to_progress(
        current_progress: float, target_progress: float, speed: PPoly, cur: float
    ) -> Optional[float]:
        try:
            speed_now = float(speed(cur))
        except (TypeError, ValueError):
            return None
        if speed_now <= 0:
            return None
        remaining = target_progress - current_progress
        if remaining <= 0:
            return cur
        return cur + remaining / speed_now

    # Main function where the calculations happen.
    # Calculates progress function and bottlenecks as described in the paper.
    def calculate(self):
        (real_data_out, data_bottleneck_list) = self.ppoly_min(
            [
                out_i(in_i)
                for (out_i, in_i) in zip(self._task.data_funcs, self._in_data_funcs)
            ]
        )

        real_out = PPoly(real_data_out.x, real_data_out.c)
        bottlenecks = data_bottleneck_list.copy()
        max_value = real_out(real_out.x[-1])
        cur = 0
        while abs(
            self.last_segment(real_out, cur)(cur) - max_value
        ) > self.precision_threshold(
            max_value
        ):  # out_cpu may not be 0, todo: maybe extend for such cases
            eps = max(
                1e-12,
                self.precision_threshold(max(1.0, abs(cur), abs(max_value))),
            )
            last_segment = self.last_segment(real_out, cur)
            next = TaskExecution.segment_end(real_out, cur)
            current_segment = real_out[cur:next]
            current_data_segment = real_data_out[cur:next]
            speedlist = [
                TaskExecution.get_speed(in_i, out_i, cur, last_segment(cur))
                for (in_i, out_i) in zip(self._in_cpu_funcs, self._task.cpu_funcs)
            ]
            speed, speed_bottleneck_list = self.ppoly_min_next_segment(
                [s[0] for s in speedlist], cur
            )  # (cycles/s) / (cylcles/b) = b/s
            speed_max_progress = min([s[1] for s in speedlist])
            speed_bottleneck_list = [-1 - x for x in speed_bottleneck_list]
            next = min(next, speed.x[1])
            ispeed = speed.antiderivative()

            gap = (current_data_segment - last_segment)(cur)
            threshold = self.precision_threshold(
                current_data_segment(cur)
            )  # todo: really think about that threshold
            gap = 0 if abs(gap) < threshold else gap
            if gap < 0:
                raise ValueError(
                    "Gap is negative, but must not be. Calculation is probably alternative correct."
                )
            elif gap != 0:
                func = ispeed - ispeed(cur) + last_segment(cur)
                roots = TaskExecution.normalize_solutions(
                    (func - current_data_segment).roots()
                )
                roots = roots[bisect.bisect_right(roots, cur) :]
                target_progress = min(max_value, speed_max_progress)
                solve_candidates = TaskExecution.normalize_solutions(
                    func.solve(target_progress)
                )
                if len(solve_candidates) == 0:
                    estimate = TaskExecution.estimate_time_to_progress(
                        last_segment(cur), target_progress, speed, cur
                    )
                    if estimate is not None:
                        solve_candidates = [estimate]
                next = min(
                    next,
                    roots[0] if len(roots) > 0 else math.inf,
                    solve_candidates[0] if len(solve_candidates) > 0 else math.inf,
                )
                if not math.isfinite(next) or next <= cur + eps:
                    fallback = TaskExecution.estimate_time_to_progress(
                        last_segment(cur), target_progress, speed, cur
                    )
                    if fallback is not None and fallback > cur + eps:
                        next = fallback
                    else:
                        if (
                            abs(last_segment(cur) - max_value)
                            <= self.precision_threshold(max_value)
                        ):
                            break
                        raise TimeoutError(
                            "Task would never finish with the given resources."
                        )
                # find index for cur in real_out.x and in func
                cur_real_out_index = bisect.bisect_right(real_out.x, cur) - 1
                if real_out.x[cur_real_out_index] != cur:
                    cur_real_out_index = cur_real_out_index + 1
                cur_func_index = bisect.bisect_right(ispeed.x, cur) - 1
                # find index for next in real_out.x and in func
                next_real_out_index = bisect.bisect_right(real_out.x, next) - 1
                next_func_index = bisect.bisect_right(ispeed.x, next) - 1
                if ispeed.x[next_func_index] != next:
                    next_func_index = next_func_index + 1
                # update bottlenecks from cur_real_out index up to next_real_out index with speed_bottleneck_list from cur_func index up to next_func index
                bottlenecks = (
                    bottlenecks[:cur_real_out_index]
                    + speed_bottleneck_list[cur_func_index:next_func_index]
                    + bottlenecks[next_real_out_index:]
                )
                real_out[cur:next] = func
                deleted_index = 0
                if cur not in real_out.x:
                    del bottlenecks[next_real_out_index]
                    deleted_index = 1
                if next not in real_out.x:
                    del bottlenecks[
                        next_real_out_index
                        + len(speed_bottleneck_list[cur_func_index:next_func_index])
                        - deleted_index
                    ]
            else:
                rod = current_segment.derivative()
                roots = TaskExecution.normalize_solutions((rod - speed).roots())
                roots = roots[bisect.bisect_right(roots, cur) :]
                next = min(
                    next,
                    roots[0] if len(roots) > 0 else math.inf,
                    real_out[cur].x[1],
                    speed[cur].x[1],
                )
                if rod(cur) > speed(cur) or (
                    rod(cur) == speed(cur) and rod.derivative()(cur) > 0
                ):
                    func = ispeed - ispeed(cur) + current_segment(cur)
                    roots = TaskExecution.normalize_solutions(
                        (func - current_data_segment).roots()
                    )
                    roots = roots[bisect.bisect_right(roots, cur) :]
                    target_progress = min(max_value, speed_max_progress)
                    solve_candidates = TaskExecution.normalize_solutions(
                        func.solve(target_progress)
                    )
                    if len(solve_candidates) == 0:
                        estimate = TaskExecution.estimate_time_to_progress(
                            current_segment(cur), target_progress, speed, cur
                        )
                        if estimate is not None:
                            solve_candidates = [estimate]
                    next = min(
                        next,
                        roots[0] if len(roots) > 0 else math.inf,
                        solve_candidates[0] if len(solve_candidates) > 0 else math.inf,
                    )
                    if not math.isfinite(next) or next <= cur + eps:
                        fallback = TaskExecution.estimate_time_to_progress(
                            current_segment(cur), target_progress, speed, cur
                        )
                        if fallback is not None and fallback > cur + eps:
                            next = fallback
                        else:
                            if (
                                abs(current_segment(cur) - max_value)
                                <= self.precision_threshold(max_value)
                            ):
                                break
                            raise TimeoutError(
                                "Task would never finish with the given resources."
                            )
                    # find index for cur in real_out.x and in func
                    cur_real_out_index = bisect.bisect_right(real_out.x, cur) - 1
                    if real_out.x[cur_real_out_index] != cur:
                        cur_real_out_index = cur_real_out_index + 1
                    cur_func_index = bisect.bisect_right(func.x, cur) - 1
                    # find index for next in real_out.x and in func
                    next_real_out_index = bisect.bisect_right(real_out.x, next) - 1
                    next_func_index = bisect.bisect_right(func.x, next) - 1
                    if func.x[next_func_index] != next:
                        next_func_index = next_func_index + 1
                    # update bottlenecks from cur_real_out index up to next_real_out index with speed_bottleneck_list from cur_func index up to next_func index
                    bottlenecks = (
                        bottlenecks[:cur_real_out_index]
                        + speed_bottleneck_list[cur_func_index:next_func_index]
                        + bottlenecks[next_real_out_index:]
                    )
                    real_out[cur:next] = func
                    deleted_index = 0
                    if cur not in real_out.x:
                        del bottlenecks[next_real_out_index]
                        deleted_index = 1
                    if next not in real_out.x:
                        del bottlenecks[
                            next_real_out_index
                            + len(speed_bottleneck_list[cur_func_index:next_func_index])
                            - deleted_index
                        ]
                elif not math.isinf(speed_max_progress):
                    roots = TaskExecution.normalize_solutions(
                        current_segment.solve(speed_max_progress)
                    )
                    if len(roots) == 0:
                        estimate = TaskExecution.estimate_time_to_progress(
                            current_segment(cur), speed_max_progress, speed, cur
                        )
                        if estimate is not None:
                            roots = [estimate]
                    roots = roots[bisect.bisect_right(roots, cur) :]
                    if len(roots) > 0:
                        next = min(next, roots[0])

            if not math.isfinite(next) or next <= cur + eps:
                raise TimeoutError("Task would never finish with the given resources.")

            cur = next

        self._real_out = real_out[:cur]
        self._bottlenecks = bottlenecks[: len(self._real_out.c[0])]

    # Internal helper function returning all indices where the value is minimal in a list.
    @staticmethod
    def argmin_in_good(liste: list) -> List[int]:
        minval = min(liste)
        return [i for (i, val) in enumerate(liste) if minval == val]

    # Internal function taking two functions and returning a new function describing the minimum of both functions.
    @staticmethod
    def ppoly_min2(func1: PPoly, func2: PPoly) -> Tuple[PPoly, List[int]]:
        minx = min(
            filter(lambda x: x != -math.inf, itertools.chain(*[func1.x, func2.x]))
        )
        maxx = max(itertools.chain(*[func1.x, func2.x]))
        res = PPoly([-math.inf, minx], [[min([func1(minx), func2(minx)])]])
        reslist = []
        diff = func1 - func2
        for root in sorted(diff.roots()):
            if root <= minx or root >= maxx or math.isnan(root) or root <= res.x[-1]:
                continue
            if diff.integrate(res.x[-1], root) < 0:  # func1 is min in this segment
                xpoly = func1[res.x[-1] : root]
                res.extend(xpoly.c, xpoly.x[1:])
                reslist = reslist + [0] * len(xpoly.c[0])
            else:
                xpoly = func2[res.x[-1] : root]
                res.extend(xpoly.c, xpoly.x[1:])
                reslist = reslist + [1] * len(xpoly.c[0])

        if diff.integrate(res.x[-1], maxx) < 0:  # func1 is min in last segment
            xpoly = func1[res.x[-1] : maxx]
            res.extend(xpoly.c, xpoly.x[1:])
            reslist = reslist + [0] * len(xpoly.c[0])
        else:
            xpoly = func2[res.x[-1] : maxx]
            res.extend(xpoly.c, xpoly.x[1:])
            reslist = reslist + [1] * len(xpoly.c[0])

        return (res[minx : res.x[-1]], reslist)

    # Same as above, different approach.
    @staticmethod
    def ppoly_min(funcs: List[PPoly]) -> Tuple[PPoly, List[int]]:
        if len(funcs) == 1:
            return funcs[0], [0] * len(funcs[0].c[0])

        (res, reslist) = TaskExecution.ppoly_min2(funcs[0], funcs[1])
        for i, f in enumerate(funcs[2:]):
            (newres, mergelist) = TaskExecution.ppoly_min2(res, f)
            # merge reslist and mergelist
            for j, x in enumerate(newres.x[0:-1]):
                if mergelist[j] == 1:
                    mergelist[j] = i + 2
                else:  # current res is min, so we need to take information which original function it was from res and reslist
                    oldsection = bisect.bisect_right(res.x, x) - 1
                    mergelist[j] = reslist[oldsection if oldsection >= 0 else 0]
            res = newres
            reslist = mergelist
        return (res, reslist)

    # From a list of functions, get for each the segment that includes `segment_start`, calculate the minimum of those single-segment functions and return it up the point the segment would change.
    @staticmethod
    def ppoly_min_next_segment(
        funcs: List[PPoly], segment_start: float
    ) -> Tuple[PPoly, List[int]]:
        func_segments = []
        for func in funcs:  # extract relevant segment starting at segment_start from func and put it into func_segments
            func_segments.append(TaskExecution.next_segment_only(func, segment_start))

        # execute regular ppoly_min on func_segments
        res, reslist = TaskExecution.ppoly_min(func_segments)
        if res.x[0] != segment_start:  # some sanity check for now
            raise ValueError("Something went wrong.")
        res.x[1] = min([f.x[1] for f in func_segments])
        return (res, reslist)

    # Returns the end of the segment of `func` that includes `start`.
    @staticmethod
    def segment_end(func: PPoly, start: float) -> float:
        idx = bisect.bisect_left(func.x, start)
        if idx == len(func.x):  # start is beyond last x
            segment_end = math.inf
        elif start == func.x[idx]:
            segment_end = func.x[idx + 1] if idx + 1 < len(func.x) else math.inf
        else:
            segment_end = func.x[idx]
        return segment_end

    # Internal function returning the segment that `start` is in up to its 'natural' end.
    @staticmethod
    def next_segment_only(func: PPoly, start: float) -> PPoly:
        return func[start : TaskExecution.segment_end(func, start)]

    # Retrieves the progress function and the list of bottlenecks calculated for this task execution.
    def get_result(self) -> Tuple[PPoly, List[int]]:
        return (self._real_out, self._bottlenecks)


# Helper function for choosing a color for a bottleneck.
def get_color_index(bottleneck: int, bottlenecks: List[int]) -> int:
    result = bottleneck - min(bottlenecks)
    return result if result >= 0 else abs(result) + max(bottlenecks)


# Plots a function `func` to the matplotlib axis `ax`.
# `colordesc` describes the color(s) the function should have. Either as one color for the whole function or as a list of bottlenecks to color the segments depending on their bottleneck.
def PlotPPoly(
    ax,
    func: PPoly,
    colordesc: Union[List[int], Any],
    colorbase: List[int] = None,
    fromtonum: slice = slice(None, None, None),
):
    if colorbase == None:
        colorbase = colordesc

    start = func.x[0] if fromtonum.start == None else fromtonum.start
    stop = func.x[-1] if fromtonum.stop == None else fromtonum.stop
    num = 1000 if fromtonum.step == None else fromtonum.step

    if type(colordesc) == list:
        if len(colordesc) != len(func.c[0]):
            raise ValueError(
                "Length of color description list must match number of function sections."
            )

        usedlabels = []
        relevantFunc = func[start:stop]
        for i, x in enumerate(relevantFunc.x[:-1]):
            tempnum = ((func.x[i + 1] - x) / (stop - start)) * num
            xs = np.linspace(x, func.x[i + 1], int(tempnum))
            colorindex = bisect.bisect_right(func.x, x)
            colorindex = (colorindex - 1) if colorindex > 0 else 0
            bottleneck = colordesc[colorindex]
            color = "C" + str(get_color_index(bottleneck, colorbase))
            label = "resource" if bottleneck < 0 else "data"
            label = label + str(bottleneck if bottleneck >= 0 else -1 - bottleneck)
            if label not in usedlabels:
                ax.plot(xs, relevantFunc(xs), color, label=label)
                usedlabels.append(label)
            else:
                ax.plot(xs, relevantFunc(xs), color)
            ax.axvspan(x, func.x[i + 1], facecolor=color, alpha=0.5)
    else:  # colordesc is single value, directly pass to plot function as color
        xs = np.linspace(start, stop, num)
        ax.plot(xs, func(xs), colordesc)


# future work:
# update in_{data,cpu} functions (with minimal recalculations)
#   may be simpler / faster if only piecewise linear functions are used
