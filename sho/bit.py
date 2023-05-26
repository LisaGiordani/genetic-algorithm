import math
import numpy as np
import copy
from SALib.sample import saltelli

from . import x,y,pb

########################################################################
# Objective functions
########################################################################

def cover_sum(sol, domain_width, sensor_range, dim):
    """Compute the coverage quality of the given array of bits."""
    assert(0 < sensor_range <= math.sqrt(2))
    assert(0 < domain_width)
    assert(dim > 0)
    assert(len(sol) >= dim)
    assert(np.sum(sol) > 0)
    domain = np.zeros((domain_width,domain_width))
    sensors = to_sensors(sol)
    cov = pb.coverage(domain, sensors, sensor_range*domain_width)
    s = np.sum(cov)
    assert(s >= len(sensors))
    return s


def to_sensors(sol):
    """Convert an square array of d lines/columns containing n ones
    to an array of n 2-tuples with related coordinates.

    >>> to_sensors([[1,0],[1,0]])
    [(0, 0), (0, 1)]
    """
    sensors = []
    for i in range(len(sol)):
        for j in range(len(sol[i])):
            if sol[i][j] == 1:
                sensors.append((j,i))
    return sensors


########################################################################
# Initialization
########################################################################

def rand(dim, domain_width, nb_sensors, sensor_range):
    """"Draw a random domain containing nb_sensors ones."""
    domain = np.zeros( (domain_width,domain_width) )
    for x,y in np.random.randint(0, domain_width, (nb_sensors, 2)):
        domain[y][x] = 1
    return domain

def rand_heuristics(dim, domain_width, nb_sensors, sensor_range):
    """Draw a quasi-random with low discrepancy domain containing nb_sensors ones 
    in a reduced domain [sensor_range/np.sqrt(2), domain_width-sensor_range/np.sqrt(2)]**dim"""
    problem = {
        'num_vars': dim,
        'bounds': [[sensor_range/np.sqrt(2), domain_width-sensor_range/np.sqrt(2)]]*dim
        }       
    vect = saltelli.sample(problem, int(np.ceil(nb_sensors/4)), calc_second_order=False)[:nb_sensors] # saltelli sampling is inspired by Sobol sampling
    
    domain = np.zeros( (domain_width,domain_width) )
    for coord in vect:
        domain[int(coord[0])][int(coord[1])] = 1
    
    return domain


########################################################################
# Neighborhood
########################################################################

def neighb_square(sol, scale, domain_width):
    """Draw a random array by moving every ones to adjacent cells."""
    assert(0 < scale <= 1)
    # Copy, because Python pass by reference
    # and we may not want to alter the original solution.
    new = copy.copy(sol)
    for py in range(len(sol)):
        for px in range(len(sol[py])):
            # Indices order is (y,x) in order to match
            # coordinates of images (row,col).
            if sol[py][px] == 1:
                # Add a one somewhere around.
                w = scale/2 * domain_width
                ny = np.random.randint(py-w,py+w)
                nx = np.random.randint(px-w,px+w)
                ny = min(max(0,ny),domain_width-1)
                nx = min(max(0,nx),domain_width-1)

                if new[ny][nx] != 1:
                    new[py][px] = 0 # Remove original position.
                    new[ny][nx] = 1
                # else pass
    return new

def neighb_generation(sol, scale, domain_width):
    """Draw a random array by moving every ones to adjacent cells
    and genrating feasible solutions if necessary"""
    assert(0 < scale <= 1)
    # Copy, because Python pass by reference
    # and we may not want to alter the original solution.
    new = copy.copy(sol)
    for py in range(len(sol)):
        for px in range(len(sol[py])):
            # Indices order is (y,x) in order to match
            # coordinates of images (row,col).
            if sol[py][px] == 1:
                # Add a one somewhere around.
                w = scale/2 * domain_width
                ny = np.random.randint(py-w,py+w+1)
                nx = np.random.randint(px-w,px+w+1)
                if ny < 0:
                    inf = 0
                    sup = min(2*py,domain_width-1)
                    if py == 0: # sous-espace de tirage réduit au singleton {0}
                        sup = min(py+2*w,domain_width-1)
                    ny = np.random.randint(inf,sup+1)
                if (ny > domain_width-1):
                    inf = max(0,2*py-domain_width)
                    sup = domain_width-1
                    if py == domain_width-1: # sous-espace de tirage réduit au singleton {domain_width-1}
                        inf = max(0,py-2*w)
                    ny = np.random.randint(inf,sup+1)
                if nx < 0:
                    inf = 0
                    sup = min(2*px,domain_width-1)
                    if px == 0: # sous-espace de tirage réduit au singleton {0}
                        sup = min(px+2*w,domain_width-1)
                    nx = np.random.randint(inf,sup+1)
                if (nx > domain_width-1):
                    inf = max(0,2*px-domain_width)
                    sup = domain_width-1
                    if px == domain_width-1: # sous-espace de tirage réduit au singleton {domain_width-1}
                        inf = max(0,px-2*w)
                    nx = np.random.randint(inf,sup+1)

                if new[ny][nx] != 1:
                    new[py][px] = 0 # Remove original position.
                    new[ny][nx] = 1
                # else pass
    return new

