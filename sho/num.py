import math
import numpy as np
from SALib.sample import saltelli

from . import pb

########################################################################
# Objective functions
########################################################################

# Decoupled from objective functions, so as to be used in display.
def to_sensors(sol):
    """Convert a vector of n*2 dimension to an array of n 2-tuples.

    >>> to_sensors([0,1,2,3])
    [(0, 1), (2, 3)]
    """
    assert(len(sol)>0)
    sensors = []
    for i in range(0,len(sol),2):
        sensors.append( ( int(math.floor(sol[i])), int(math.floor(sol[i+1])) ) )
    return sensors


def cover_sum(sol, domain_width, sensor_range, dim):
    """Compute the coverage quality of the given vector."""
    assert(0 < sensor_range <= domain_width * math.sqrt(2))
    assert(0 < domain_width)
    assert(dim > 0)
    assert(len(sol) >= dim)
    domain = np.zeros((domain_width,domain_width))
    sensors = to_sensors(sol)
    cov = pb.coverage(domain, sensors, sensor_range*domain_width)
    s = np.sum(cov)
    assert(s >= len(sensors))
    return s


########################################################################
# Initialization
########################################################################

def rand(dim, nb_sensors, scale, sensor_range):
    """Draw a random vector in [0,scale]**dim."""
    return np.random.random(dim*nb_sensors) * scale

def rand_heuristics(dim, nb_sensors, scale, sensor_range):
    """Draw a random vector in [sensor_range/sqrt(2),scale-sensor_range/sqrt(2)]**dim with low discrepancy"""
    problem = {
        'num_vars': dim,
        'bounds': [[sensor_range/np.sqrt(2), scale-sensor_range/np.sqrt(2)]]*dim
        }
    vect = saltelli.sample(problem, int(np.ceil(nb_sensors/4)), calc_second_order=False)[:nb_sensors] # saltelli sampling is inspired by Sobol sampling
    return vect.flatten()


########################################################################
# Neighborhood
########################################################################

def neighb_square(sol, scale, domain_width):
    """Draw a random vector in a square of witdh `scale` in [0,1]
    as a fraction of the domain width around the given solution."""
    assert(0 < scale <= 1)
    side = domain_width * scale;
    new = sol + (np.random.random(len(sol)) * side - side/2)
    return new

def neighb_generation(sol, scale, domain_width):
    """Draw a random vector in a square of witdh `scale` in [0,1]
    as a fraction of the domain width around the given solution
    with generation of feasible solutions if necessary."""
    assert(0 < scale <= 1)
    side = domain_width * scale;
    new = sol + (np.random.random(len(sol)) * side - side/2)
    i = 0
    for coor in new:
        if coor < 0:
            if sol[i] != 0:
                new[i] = np.random.uniform(0,min(2*sol[i],domain_width))
            else:
                new[i] = np.random.uniform(0,min(side,domain_width))
        if coor > domain_width:
            if sol[i] != domain_width:
                new[i] = np.random.uniform(max(0,2*sol[i]-domain_width),domain_width)
            else:
                new[i] = np.random.uniform(max(0,domain_width-side),domain_width)
        i += 1
    return new