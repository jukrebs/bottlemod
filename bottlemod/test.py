import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from bottlemod.func import Func
from bottlemod.ppoly import PPoly
from bottlemod.task import PlotPPoly, Task, TaskExecution, get_color_index

out_cpu = [PPoly([0, 10000], [[2]])]
in_cpu = [PPoly([0, 10000], [[1]])]
out_data = [Func([0, 10000], [[1, 0]])]
in_data = [Func([0, 10000], [[1, 0]])]


# sanity adjustments
for oc in out_cpu:
    oc.x[-1] = out_data[0](out_data[0].x[-1])

endresult, finalbottlenecks = TaskExecution(
    Task(out_cpu, out_data), in_cpu, in_data
).get_result()

print(endresult)
print(finalbottlenecks)


# potential output progress by storable resource inputs only
fig = plt.figure()
(result, bottlenecks) = TaskExecution.ppoly_min(
    [out_i(in_i) for (out_i, in_i) in zip(out_data, in_data)]
)
mpl.rc("lines", linestyle="solid")
PlotPPoly(plt, result, bottlenecks, finalbottlenecks)
mpl.rc("lines", linestyle="--")
for i, f in enumerate([out_i(in_i) for (out_i, in_i) in zip(out_data, in_data)]):
    # print(i, f)
    xs = np.linspace(f.x[0], f.x[-1], 1000)
    color = "C" + str(get_color_index(i, finalbottlenecks))
    plt.plot(xs, f(xs), color, alpha=0.7)
plt.axis((result.x[0], result.x[-1], result(result.x[0]), result(result.x[-1]) * 1.02))
plt.legend()
plt.xlabel("time [time units]")
plt.ylabel("potential progress [%]")
w, h = fig.get_size_inches()
fig.set_size_inches(w * 1, h * 0.4)
plt.savefig("dataprogress.pdf", bbox_inches="tight", pad_inches=0)


fig, (ax1, ax2) = plt.subplots(2, 1)
mpl.rc("lines", linestyle="solid")
PlotPPoly(ax1, result, bottlenecks)
print("")
mpl.rc("lines", linestyle="--")
for i, f in enumerate([out_i(in_i) for (out_i, in_i) in zip(out_data, in_data)]):
    xs = np.linspace(f.x[0], f.x[-1], 1000)
    color = "C" + str(get_color_index(i, bottlenecks))
    ax1.plot(xs, f(xs), color, alpha=0.7)
ax1.axis((result.x[0], result.x[-1], result(result.x[0]), result(result.x[-1]) * 1.02))
ax1.set_ylabel("progress [%]")
ax1.tick_params(
    axis="both",
    which="both",
    bottom=True,
    top=False,
    labelbottom=False,
    left=False,
    right=True,
    labelleft=False,
    labelright=True,
)

dresult = result.derivative()
mpl.rc("lines", linestyle="solid")
xs = np.linspace(result.x[0], result.x[-1], 1000)
for i, (inc, outc) in enumerate(zip(in_cpu, out_cpu)):
    color = "C" + str(get_color_index(-1 - i, bottlenecks))
    mpl.rc("lines", linestyle="--")
    ax2.plot(xs, inc(xs) * 100 / 6.5, color)
    mpl.rc("lines", linestyle="solid")
    ax2.plot(
        xs,
        [
            dresult(x) * outc(result)(x) * 100 / 6.5 + (i if x >= 33 and x < 50 else 0)
            for x in xs
        ],
        color,
        label="resource" + str(i),
    )
_, _, _, ymax = ax2.axis()
ax2.axis((result.x[0], result.x[-1], 0, ymax))
ax2.set_ylabel("resources [%]")
ax2.tick_params(
    axis="both",
    which="both",
    bottom=True,
    top=False,
    left=False,
    labelbottom=False,
    labelleft=False,
    labelright=True,
    right=True,
)
plt.xlabel("time [time units]")
w, h = fig.get_size_inches()
# fig.set_size_inches(w * 1.75, h * 1.2)
fig.set_size_inches(w * 1.75, h * 0.8)
handles, labels = ax1.get_legend_handles_labels()
labels, handles = zip(
    *sorted(zip(labels, handles), key=lambda t: t[0])
)  # sort both labels and handles by labels
ax1.legend(
    handles,
    labels,
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc="lower left",
    mode="expand",
    ncol=6,
)
plt.savefig("finalprogressandmore.pdf", bbox_inches="tight", pad_inches=0.1)
