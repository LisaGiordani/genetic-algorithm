import numpy as np
import matplotlib.pyplot as plt

from sho import *

def yonly_cover_sum(sol, domain_width, sensor_range, fixed_x = (10,30)):
    """Compute the coverage quality of the given vector."""
    domain = np.zeros((domain_width,domain_width))
    sensors = [ (fixed_x[0], sol[0]), (fixed_x[1], sol[1]) ]
    return np.sum(pb.coverage(domain, sensors, sensor_range))


if __name__ == "__main__":

    d = 2
    w = 40
    n = 2
    r = 0.3 * w

    # Common termination and checkpointing.
    history = []
    iters = make.iter(
                iters.several,
                agains = [
                    make.iter(iters.max,
                        nb_it = 100),
                    make.iter(iters.log,
                        fmt="\r{it} {val}"),
                    make.iter(iters.history,
                        history = history)
                ]
            )

    x0,x1 = 0.25*w, 0.75*w

    val,sol = algo.greedy(
            make.func(yonly_cover_sum,
                domain_width = w,
                sensor_range = r,
                fixed_x = (x0,x1) ),
            make.init(num.rand,
                dim = 2, # Two sensors moving along y axis.
                scale = 1.0),
            make.neig(num.neighb_square,
                scale = 0.1,
                domain_width = w
                ),
            iters
        )
    sensors = [ (int(x0), int(round(sol[0]))), (int(x1), int(round(sol[1]))) ]

    print("\n{} : {}".format(val,sensors))

    shape=(w,w)
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    f = make.func(yonly_cover_sum,
                    domain_width = w,
                    sensor_range = r)
    plot.surface(ax1, shape, f)
    plot.path(ax1, shape, history)

    domain = np.zeros(shape)
    domain = pb.coverage(domain, sensors, r)
    domain = plot.highlight_sensors(domain, sensors)
    ax2.imshow(domain)

    plt.show()
