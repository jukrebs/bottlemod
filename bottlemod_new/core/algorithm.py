"""
Core Bottleneck Analysis Algorithm

This module contains the pure algorithmic functions for bottleneck analysis.
It implements a direct-construction algorithm that builds the progress curve
segment by segment, identifying the limiting factor (data or a specific resource)
at each step. This approach is an efficient alternative to the iterative refinement
described in the original BottleMod paper.
"""

import bisect
import math
from typing import List, Tuple

from .func import Func
from .ppoly import PPoly


def bottleneck_analysis(
    resource_requirements: List[PPoly],
    resource_inputs: List[PPoly],
    data_dependencies: List[Func],
    data_inputs: List[Func],
) -> Tuple[PPoly, List[int]]:
    """
    Core bottleneck analysis algorithm.

    Analyzes task execution to determine:
    1. The progress function P(t), which describes how much progress is made over time.
    2. A timeline of bottlenecks, indicating which resource or data dependency limits
       progress at each moment.

    The algorithm works by constructing the progress curve `P(t)` piece by piece,
    rather than using iterative refinement. In each segment, it determines whether
    progress is limited by data availability or by the rate of a resource.

    Parameters
    ----------
    resource_requirements : List[PPoly]
        A list of cumulative resource requirement functions, R_R,l(p). Each function
        maps progress `p` to the total amount of resource `l` needed to reach it.
    resource_inputs : List[PPoly]
        A list of cumulative resource availability functions, I_R,l(t). Each function
        maps time `t` to the total amount of resource `l` that has been made available.
        The algorithm uses the derivative of these functions to determine the resource rate.
    data_dependencies : List[Func]
        A list of data requirement functions, R_D,k(p). Each function maps progress `p`
        to the amount of data input `k` required.
    data_inputs : List[Func]
        A list of data availability functions, I_D,k(t). Each function maps time `t`
        to the amount of data from input `k` that is available.

    Returns
    -------
    Tuple[PPoly, List[int]]
        - A `PPoly` instance representing the progress function P(t).
        - A list of bottleneck indices. A negative value indicates a data dependency
          (e.g., -1 for the first data dependency), while a non-negative value
          corresponds to the index of the limiting resource in `resource_requirements`.

    Notes
    -----
    This is a direct-construction implementation of the concepts from the BottleMod
    paper (ICPE'25), optimized for performance with piecewise polynomials.
    """
    # Step 1: Calculate data-limited progress
    # Progress cannot exceed what data is available.
    if (data_dependencies and not data_inputs) or (
        not data_dependencies and data_inputs
    ):
        raise ValueError(
            "`data_dependencies` and `data_inputs` must be provided together."
        )

    if data_dependencies and data_inputs:
        # P_D,k(t) = R_D,k(I_D,k(t))
        data_progress_functions = [
            req(avail) for (req, avail) in zip(data_dependencies, data_inputs)
        ]
        # P_D(t) = min_k(P_D,k(t))
        real_data_out, data_bottleneck_list = ppoly_min(data_progress_functions)
    else:
        # No data dependencies - data is immediately available
        # Create identity function: at time t, progress t is available
        if resource_requirements:
            max_progress = resource_requirements[0].x[-1]
        else:
            max_progress = 1.0
        # Data progresses at rate 1 (identity: progress = time)
        # PPoly format: c is list of coefficient lists, one per segment
        # For linear f(t) = t, we need coefficient [1, 0] for segment [0, max_progress]
        real_data_out = PPoly([0, max_progress], [[1], [0]])
        data_bottleneck_list = []

    # Step 2: Apply resource constraints
    real_out = PPoly(real_data_out.x, real_data_out.c)
    bottlenecks = data_bottleneck_list.copy()
    max_value = real_out(real_out.x[-1])
    cur = 0

    def precision_threshold(val: float) -> float:
        return 40 * val * 10 ** (-15)

    while abs(last_segment(real_out, cur)(cur) - max_value) > precision_threshold(
        max_value
    ):
        last_seg = last_segment(real_out, cur)
        next_t = segment_end(real_out, cur)
        current_segment = real_out[cur:next_t]
        current_data_segment = real_data_out[cur:next_t]

        # Calculate allowed speeds from each resource
        speedlist = [
            get_speed(in_func, out_func, cur, last_seg(cur))
            for (in_func, out_func) in zip(resource_inputs, resource_requirements)
        ]

        # Find minimum speed (bottleneck)
        speed, speed_bottleneck_list = ppoly_min_next_segment(
            [s[0] for s in speedlist], cur
        )
        speed_max_progress = min([s[1] for s in speedlist])
        speed_bottleneck_list = [-1 - x for x in speed_bottleneck_list]

        next_t = min(next_t, speed.x[1])
        ispeed = speed.antiderivative()

        # Check if we need to catch up with data
        gap = (current_data_segment - last_seg)(cur)
        threshold = precision_threshold(current_data_segment(cur))
        gap = 0 if abs(gap) < threshold else gap

        if gap < 0:
            raise ValueError(
                "Gap is negative, but must not be. Calculation is probably incorrect."
            )
        elif gap != 0:
            # Data-limited: integrate speed until we catch up
            func = ispeed - ispeed(cur) + last_seg(cur)
            roots = (func - current_data_segment).roots()
            roots = roots[bisect.bisect_right(roots, cur) :]
            solve_result = func.solve(min(max_value, speed_max_progress))
            solve_result = solve_result[bisect.bisect_right(solve_result, cur) :]
            next_t = min(
                next_t,
                roots[0] if len(roots) > 0 else math.inf,
                solve_result[0] if len(solve_result) > 0 else math.inf,
            )

            # Update real_out and bottlenecks
            cur_real_out_index = bisect.bisect_right(real_out.x, cur) - 1
            if real_out.x[cur_real_out_index] != cur:
                cur_real_out_index = cur_real_out_index + 1
            cur_func_index = bisect.bisect_right(ispeed.x, cur) - 1

            next_real_out_index = bisect.bisect_right(real_out.x, next_t) - 1
            next_func_index = bisect.bisect_right(ispeed.x, next_t) - 1
            if ispeed.x[next_func_index] != next_t:
                next_func_index = next_func_index + 1

            bottlenecks = (
                bottlenecks[:cur_real_out_index]
                + speed_bottleneck_list[cur_func_index:next_func_index]
                + bottlenecks[next_real_out_index:]
            )
            real_out[cur:next_t] = func

            deleted_index = 0
            if cur not in real_out.x:
                if 0 <= next_real_out_index < len(bottlenecks):
                    del bottlenecks[next_real_out_index]
                    deleted_index = 1
            if next_t not in real_out.x:
                del_idx = (
                    next_real_out_index
                    + len(speed_bottleneck_list[cur_func_index:next_func_index])
                    - deleted_index
                )
                if 0 <= del_idx < len(bottlenecks):
                    del bottlenecks[del_idx]
        else:
            # No gap: check if resource-limited or data-limited
            rod = current_segment.derivative()
            roots = (rod - speed).roots()
            roots = roots[bisect.bisect_right(roots, cur) :]
            next_t = min(
                next_t,
                roots[0] if len(roots) > 0 else math.inf,
                real_out[cur].x[1],
                speed[cur].x[1],
            )

            if rod(cur) > speed(cur) or (
                rod(cur) == speed(cur) and rod.derivative()(cur) > 0
            ):
                # Resource-limited
                func = ispeed - ispeed(cur) + current_segment(cur)
                roots = (func - current_data_segment).roots()
                roots = roots[bisect.bisect_right(roots, cur) :]
                solve_result = func.solve(min(max_value, speed_max_progress))
                solve_result = solve_result[bisect.bisect_right(solve_result, cur) :]
                next_t = min(
                    next_t,
                    roots[0] if len(roots) > 0 else math.inf,
                    solve_result[0] if len(solve_result) > 0 else math.inf,
                )

                cur_real_out_index = bisect.bisect_right(real_out.x, cur) - 1
                if real_out.x[cur_real_out_index] != cur:
                    cur_real_out_index = cur_real_out_index + 1
                cur_func_index = bisect.bisect_right(func.x, cur) - 1

                next_real_out_index = bisect.bisect_right(real_out.x, next_t) - 1
                next_func_index = bisect.bisect_right(func.x, next_t) - 1
                if func.x[next_func_index] != next_t:
                    next_func_index = next_func_index + 1

                bottlenecks = (
                    bottlenecks[:cur_real_out_index]
                    + speed_bottleneck_list[cur_func_index:next_func_index]
                    + bottlenecks[next_real_out_index:]
                )
                real_out[cur:next_t] = func

                deleted_index = 0
                if cur not in real_out.x:
                    if 0 <= next_real_out_index < len(bottlenecks):
                        del bottlenecks[next_real_out_index]
                        deleted_index = 1
                if next_t not in real_out.x:
                    del_idx = (
                        next_real_out_index
                        + len(speed_bottleneck_list[cur_func_index:next_func_index])
                        - deleted_index
                    )
                    if 0 <= del_idx < len(bottlenecks):
                        del bottlenecks[del_idx]
            elif not math.isinf(speed_max_progress):
                solve_roots = current_segment.solve(speed_max_progress)
                roots = solve_roots[bisect.bisect_right(solve_roots, cur) :]
                if len(roots) > 0:
                    next_t = min(next_t, roots[0])

        if next_t == cur:
            raise TimeoutError("Task would never finish with the given resources.")

        cur = next_t

    # Trim to actual completion
    real_out = real_out[:cur]
    bottlenecks = bottlenecks[: len(real_out.c[0])]

    return real_out, bottlenecks


def get_speed(
    in_function: PPoly, out_function: PPoly, t: float, progress: float
) -> Tuple[PPoly, float]:
    """
    Calculate the speed a resource allows at time t.

    Speed = available_resource_rate / required_resource_rate

    Parameters
    ----------
    in_function : PPoly
        Resource input function I_l(t)
    out_function : PPoly
        Resource requirement function R_l(p)
    t : float
        Current time
    progress : float
        Current progress

    Returns
    -------
    Tuple[PPoly, float]
        - Speed function (progress per unit time)
        - Maximum progress this speed is valid for
    """
    in_function = next_segment_only(in_function, t)
    segment_end_t = in_function.x[-1]
    if segment_end_t <= t:
        segment_end_t = t + 1e-12

    relevant_out_function = out_function[progress]
    max_progress_validity = relevant_out_function.x[-1]

    # Take derivative to convert cumulative to rate
    in_function_derivative = in_function.derivative()
    in_function_derivative.x[0], in_function_derivative.x[1] = -math.inf, math.inf

    out_function_derivative = relevant_out_function.derivative()
    out_function_derivative.x[0], out_function_derivative.x[1] = -math.inf, math.inf

    try:
        result = in_function_derivative / out_function_derivative
    except ValueError as exc:
        raise ValueError(
            f"Unable to compute speed at t={t}: input domain {in_function_derivative.x}, output domain {out_function_derivative.x}"
        ) from exc

    result = result[t:segment_end_t]
    return result, max_progress_validity


def ppoly_min(funcs: List[PPoly]) -> Tuple[PPoly, List[int]]:
    """
    Find the minimum across multiple PPoly functions.

    Parameters
    ----------
    funcs : List[PPoly]
        List of piecewise polynomial functions

    Returns
    -------
    Tuple[PPoly, List[int]]
        - Minimum function
        - List of indices indicating which function is minimum in each segment
    """
    if len(funcs) == 1:
        return funcs[0], [0] * len(funcs[0].c[0])

    (res, reslist) = ppoly_min2(funcs[0], funcs[1])
    for i, f in enumerate(funcs[2:]):
        (newres, mergelist) = ppoly_min2(res, f)
        # Merge reslist and mergelist
        for j, x in enumerate(newres.x[0:-1]):
            if mergelist[j] == 1:
                mergelist[j] = i + 2
            else:
                oldsection = bisect.bisect_right(res.x, x) - 1
                mergelist[j] = reslist[oldsection if oldsection >= 0 else 0]
        res = newres
        reslist = mergelist
    return (res, reslist)


def ppoly_min2(func1: PPoly, func2: PPoly) -> Tuple[PPoly, List[int]]:
    """
    Find minimum of two PPoly functions.

    Parameters
    ----------
    func1 : PPoly
        First function
    func2 : PPoly
        Second function

    Returns
    -------
    Tuple[PPoly, List[int]]
        - Minimum function
        - List of indices (0 = func1 is min, 1 = func2 is min)
    """
    import itertools

    minx = min(filter(lambda x: x != -math.inf, itertools.chain(*[func1.x, func2.x])))
    maxx = max(itertools.chain(*[func1.x, func2.x]))
    res = PPoly([-math.inf, minx], [[0], [min([func1(minx), func2(minx)])]])
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


def ppoly_min_next_segment(
    funcs: List[PPoly], segment_start: float
) -> Tuple[PPoly, List[int]]:
    """
    Find minimum for the next segment only.

    Parameters
    ----------
    funcs : List[PPoly]
        List of functions
    segment_start : float
        Start of segment

    Returns
    -------
    Tuple[PPoly, List[int]]
        - Minimum function for next segment
        - Bottleneck indices
    """
    func_segments = []
    for func in funcs:
        func_segments.append(next_segment_only(func, segment_start))

    res, reslist = ppoly_min(func_segments)
    if res.x[0] != segment_start:
        raise ValueError("Something went wrong in ppoly_min_next_segment.")

    segment_end = min(f.x[1] for f in func_segments)
    res = res[segment_start:segment_end]
    reslist = reslist[: len(res.c[0])]
    return (res, reslist)


def segment_end(func: PPoly, start: float) -> float:
    """
    Get the end of the segment containing start.

    Parameters
    ----------
    func : PPoly
        Function to query
    start : float
        Point to find segment for

    Returns
    -------
    float
        End of the segment
    """
    idx = bisect.bisect_left(func.x, start)
    if idx == len(func.x):
        segment_end = math.inf
    elif start == func.x[idx]:
        segment_end = func.x[idx + 1] if idx + 1 < len(func.x) else math.inf
    else:
        segment_end = func.x[idx]
    return segment_end


def next_segment_only(func: PPoly, start: float) -> PPoly:
    """
    Extract the segment containing start up to its natural end.

    Parameters
    ----------
    func : PPoly
        Function to extract from
    start : float
        Start of segment

    Returns
    -------
    PPoly
        The segment
    """
    return func[start : segment_end(func, start)]


def last_segment(func: PPoly, x: float) -> PPoly:
    """
    Get the last segment of a function before or at x.

    Parameters
    ----------
    func : PPoly
        Function to query
    x : float
        Point to find segment for

    Returns
    -------
    PPoly
        The last segment
    """
    last_section_index = bisect.bisect_left(func.x, x) - 1
    return (
        func[func.x[last_section_index]]
        if last_section_index >= 0
        else PPoly([0, 1], [[0], [0]])
    )
