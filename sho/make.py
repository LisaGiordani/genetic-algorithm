"""Wrappers that captures parameters of a function
and returns an operator with a given interface."""

def func(cover, **kwargs):
    """Make an objective function from the given function.
    An objective function takes a solution and returns a scalar."""
    def f(sol):
        return cover(sol,**kwargs)
    return f


def init(init, **kwargs):
    """Make an initialization operator from the given function.
    An init. op. returns a solution."""
    def f():
        return init(**kwargs)
    return f


def neig(neighb, **kwargs):
    """Make an neighborhood operator from the given function.
    A neighb. op. takes a solution and returns another one."""
    def f(sol):
        return neighb(sol, **kwargs)
    return f


def iter(iters, **kwargs):
    """Make an iterations operator from the given function.
    A iter. op. takes a value and a solution and returns
    the current number of iterations."""
    def f(i, val, sol):
        return iters(i, val, sol, **kwargs)
    return f

