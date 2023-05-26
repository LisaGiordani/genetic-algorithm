#encoding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt

from sho import make, algo, iters, plot, num, bit, pb

########################################################################
# Interface
########################################################################

if __name__=="__main__":
    import argparse

    # Dimension of the search space.
    d = 2

    can = argparse.ArgumentParser()

    can.add_argument("-n", "--nb-sensors", metavar="NB", default=10, type=int,
            help="Number of sensors")

    can.add_argument("-r", "--sensor-range", metavar="RATIO", default=0.15, type=float,
            help="Sensors' range (as a fraction of domain width, max is √2)")

    can.add_argument("-w", "--domain-width", metavar="NB", default=30, type=int,
            help="Domain width (a number of cells). If you change this you will probably need to update `--target` accordingly")

    can.add_argument("-i", "--iters", metavar="NB", default=100, type=int,
            help="Maximum number of iterations")

    can.add_argument("-s", "--seed", metavar="VAL", default=None, type=int,
            help="Random pseudo-generator seed (none for current epoch)")

    solvers = ["num_greedy","bit_greedy",
               "num_recuit_simule","bit_recuit_simule",
               "num_evolution_strategies","bit_evolution_strategies"]
    can.add_argument("-m", "--solver", metavar="NAME", default='num_recuit_simule', choices=solvers,
            help="Solver to use, among: "+", ".join(solvers))

    can.add_argument("-t", "--target", metavar="VAL", default=30*30, type=float,
            help="Objective function value target")

    can.add_argument("-y", "--steady-delta", metavar="NB", default=50, type=float,
            help="Stop if no improvement after NB iterations")

    can.add_argument("-e", "--steady-epsilon", metavar="DVAL", default=0, type=float,
            help="Stop if the improvement of the objective function value is lesser than DVAL")

    can.add_argument("-a", "--variation-scale", metavar="RATIO", default=0.3, type=float,
            help="Scale of the variation operators (as a ration of the domain width)")

    can.add_argument("-Tinit", "--temperature_init", metavar="NB", default=50, type=float,
            help="Initial value of the temperature parameter of the simulated annealing")

    can.add_argument("-b", "--beta", metavar="NB", default=5, type=float,
            help="Beta parameter of the simulated annealing (decrease speed of temperature)")

    can.add_argument("-p", "--nb_pop", metavar="NB", default=6, type=float,
            help="Number of elements in the population of the genetic algorithm")

    the = can.parse_args()

    # Minimum checks.
    assert(0 < the.nb_sensors)
    assert(0 < the.sensor_range <= math.sqrt(2))
    assert(0 < the.domain_width)
    assert(0 < the.iters)

    # Do not forget the seed option,
    # in case you would start "runs" in parallel.
    np.random.seed(the.seed)

    # Weird numpy way to ensure single line print of array.
    np.set_printoptions(linewidth = np.inf)


    # Common termination and checkpointing.
    history = []
    iters = make.iter(
                iters.several,
                agains = [
                    make.iter(iters.max,
                        nb_it = the.iters),
                    make.iter(iters.save,
                        filename = the.solver+".csv",
                        fmt = "{it} ; {val} ; {sol}\n"),
                    make.iter(iters.log,
                        fmt="\r{it} {val}"),
                    make.iter(iters.history,
                        history = history),
                    make.iter(iters.target,
                        target = the.target),
                    iters.steady(the.steady_delta, the.steady_epsilon)
                ]
            )

    # Erase the previous file.
    with open(the.solver+".csv", 'w') as fd:
        fd.write("# {} {}\n".format(the.solver,the.domain_width))

    val,sol,sensors = None,None,None

    if the.solver == "num_greedy":
        val,sol = algo.greedy(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(num.rand_heuristics,
                    dim = d,
                    nb_sensors = the.nb_sensors,
                    scale = the.domain_width,
                    sensor_range = the.sensor_range),
                make.neig(num.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_greedy":
        val,sol = algo.greedy(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(bit.rand_heuristics,
                    dim = d,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors,
                    sensor_range = the.sensor_range),
                make.neig(bit.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters
            )
        sensors = bit.to_sensors(sol)

    elif the.solver == "num_recuit_simule":
        val,sol = algo.recuit_simule(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(num.rand_heuristics,
                    dim = d,
                    nb_sensors = the.nb_sensors,
                    scale = the.domain_width,
                    sensor_range = the.sensor_range),
                make.neig(num.neighb_generation,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                the.temperature_init,
                the.beta
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_recuit_simule":
        val,sol = algo.recuit_simule(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(bit.rand_heuristics,
                    dim = d,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors,
                    sensor_range = the.sensor_range),
                make.neig(bit.neighb_generation,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                the.temperature_init,
                the.beta
            )
        sensors = bit.to_sensors(sol)

    elif the.solver == "num_evolution_strategies":
        val,sol = algo.evolution_strategies(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(num.rand_heuristics,
                    dim = d,
                    nb_sensors = the.nb_sensors,
                    scale = the.domain_width,
                    sensor_range = the.sensor_range),
                make.neig(num.neighb_generation,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                the.nb_pop
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_evolution_strategies":
        val,sol = algo.evolution_strategies(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(bit.rand_heuristics,
                    dim = d,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors,
                    sensor_range = the.sensor_range),
                make.neig(bit.neighb_generation,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                the.nb_pop
            )
        sensors = bit.to_sensors(sol)

    # Fancy output.
    print("\n{} : {}".format(val,sensors))

    shape=(the.domain_width, the.domain_width)

    fig = plt.figure()

    if the.nb_sensors ==1 and the.domain_width <= 50:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        f = make.func(num.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range * the.domain_width)
        plot.surface(ax1, shape, f)
        plot.path(ax1, shape, history)
    else:
        ax2=fig.add_subplot(111)

    domain = np.zeros(shape)
    domain = pb.coverage(domain, sensors,
            the.sensor_range * the.domain_width)
    domain = plot.highlight_sensors(domain, sensors)
    ax2.imshow(domain)

    plt.show()


def run(param):

    from sho import make, algo, iters, plot, num, bit, pb

    nb_sensors = param['nb_sensors']
    sensor_range = param['sensor_range']
    domain_width = param['domain_width']
    iters_max = param['iters_max']
    seed =  param['seed']
    solver = param['algo']
    target = param['target']
    steady_delta = param['steady_delta']
    steady_epsilon = param['steady_epsilon']
    variation_scale = param['variation_scale']
    temperature_init = param['temperature_init']
    beta = param['beta']
    nb_pop = param['nb_pop']

    # Dimension of the search space
    d = 2

    # Minimum checks.
    assert(0 < nb_sensors)
    assert(0 < sensor_range <= math.sqrt(2))
    assert(0 < domain_width)
    assert(0 < iters_max)

    # Do not forget the seed option,
    # in case you would start "runs" in parallel.
    np.random.seed(seed)

    # Weird numpy way to ensure single line print of array.
    np.set_printoptions(linewidth = np.inf)

    # Common termination and checkpointing.
    history = []
    iters = make.iter(
                iters.several,
                agains = [
                    make.iter(iters.max,
                        nb_it = iters_max),
                    make.iter(iters.save,
                        filename = solver+".csv",
                        fmt = "{it} ; {val} ; {sol}\n"),
                    make.iter(iters.log,
                        fmt="\r{it} {val}"),
                    make.iter(iters.history,
                        history = history),
                    make.iter(iters.target,
                        target = target),
                    iters.steady(steady_delta, steady_epsilon)
                ]
            )

    # Erase the previous file.
    with open(solver+".csv", 'w') as fd:
        fd.write("# {} {}\n".format(solver,domain_width))

    val,sol,sensors = None,None,None

    if solver == "num_greedy":
        val,sol = algo.greedy(
                make.func(num.cover_sum,
                    domain_width = domain_width,
                    sensor_range = sensor_range,
                    dim = d * nb_sensors),
                make.init(num.rand_heuristics,
                    dim = d,
                    nb_sensors = nb_sensors,
                    scale = domain_width,
                    sensor_range = sensor_range),
                make.neig(num.neighb_square,
                    scale = variation_scale,
                    domain_width = domain_width),
                iters
            )
        sensors = num.to_sensors(sol)

    elif solver == "bit_greedy":
        val,sol = algo.greedy(
                make.func(bit.cover_sum,
                    domain_width = domain_width,
                    sensor_range = sensor_range,
                    dim = d * nb_sensors),
                make.init(bit.rand_heuristics,
                    dim = d,
                    domain_width = domain_width,
                    nb_sensors = nb_sensors,
                    sensor_range = sensor_range),
                make.neig(bit.neighb_square,
                    scale = variation_scale,
                    domain_width = domain_width),
                iters
            )
        sensors = bit.to_sensors(sol)

    elif solver == "num_recuit_simule":
        val,sol = algo.recuit_simule(
                make.func(num.cover_sum,
                    domain_width = domain_width,
                    sensor_range = sensor_range,
                    dim = d * nb_sensors),
                make.init(num.rand_heuristics,
                    dim = d,
                    nb_sensors = nb_sensors,
                    scale = domain_width,
                    sensor_range = sensor_range),
                make.neig(num.neighb_generation,
                    scale = variation_scale,
                    domain_width = domain_width),
                iters,
                temperature_init,
                beta
            )
        sensors = num.to_sensors(sol)

    elif solver == "bit_recuit_simule":
        val,sol = algo.recuit_simule(
                make.func(bit.cover_sum,
                    domain_width = domain_width,
                    sensor_range = sensor_range,
                    dim = d * nb_sensors),
                make.init(bit.rand_heuristics,
                    dim = d,
                    domain_width = domain_width,
                    nb_sensors = nb_sensors,
                    sensor_range = sensor_range),
                make.neig(bit.neighb_generation,
                    scale = variation_scale,
                    domain_width = domain_width),
                iters,
                temperature_init,
                beta
            )
        sensors = bit.to_sensors(sol)

    elif solver == "num_evolution_strategies":
        val,sol = algo.evolution_strategies(
                make.func(num.cover_sum,
                    domain_width = domain_width,
                    sensor_range = sensor_range,
                    dim = d * nb_sensors),
                make.init(num.rand_heuristics,
                    dim = d,
                    nb_sensors = nb_sensors,
                    scale = domain_width,
                    sensor_range = sensor_range),
                make.neig(num.neighb_generation,
                    scale = variation_scale,
                    domain_width = domain_width),
                iters,
                nb_pop
            )
        sensors = num.to_sensors(sol)

    elif solver == "bit_evolution_strategies":
        val,sol = algo.evolution_strategies(
                make.func(bit.cover_sum,
                    domain_width = domain_width,
                    sensor_range = sensor_range,
                    dim = d * nb_sensors),
                make.init(bit.rand_heuristics,
                    dim = d,
                    domain_width = domain_width,
                    nb_sensors = nb_sensors,
                    sensor_range = sensor_range),
                make.neig(bit.neighb_generation,
                    scale = variation_scale,
                    domain_width = domain_width),
                iters,
                nb_pop
            )
        sensors = bit.to_sensors(sol)



