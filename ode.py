########################################################################
# MSU Hollow Earth Society: 
# Joe Epley, Elias Taira, Erin Syerson, Michael Bellaver
# AST 304, Fall 2020
# Michigan State University
########################################################################

"""
This module sets up each of the 3 ODEs (forward euler, 2nd-order Runge-Kutta, 
and 4th-order Runge-Kutta). It does this by defining 3 separate functions to 
determine what each ODE does when it is calculated.

"""

# all routines that take a single step should have the same interface
# fEuler is complete, except for documentation. you can use this as a pattern 
# for the other two routines.
def fEuler(f,t,z,h,args=()):
    """
    This sets up the forward Euler ODE. It defines it as a function that takes 
    in a function (f), time(t), position (z), time-step (h), and other arguments 
    (args). This function returns the new position (new_z) as z+h*f(t,z,*args).
    
    Arguments
        f(t,z,...)
            function that contains the RHS of the equation dz/dt = f(t,z,...)
    
        t
            the time at which the system is currently at. Only impacts anything
            if f() has a time dependence
        
        h
            the timestep between each run of the method, smaller timesteps will
            result in more accurate results, but increase computation time
    
        args (tuple, optional)
            additional arguments to pass to f
            
    Returns
        znew = z(t+h)
    """
    
    # The following trick allows us to pass additional parameters to f
    # first we make sure that args is of type tuple; if not, we make it into
    # that form
    if not isinstance(args,tuple):
        args = (args,)
    
    # when we call f, we use *args to pass it as a list of parameters.
    # for example, if elsewhere we define f like
    # def f(t,z,x,y):
    #    ...
    # then we would call this routine as
    # znew = fEuler(f,t,z,h,args=(x,y))
    #
    return z + h*f(t,z,*args)

# You will need to flesh out the following routines for a second-order
# Runge-Kutta step and a fourth order Runge-Kutta step.

def rk2(f,t,z,h,args=()):
    """
    This sets up the 2nd-order Runge-Kutta by defining it as a function which 
    takes in a function (f), time (t), position (z), time-step (h), and other 
    arguments (args). Inside the function, initial acceleration is calculated 
    (k1z) which is used to find the new acceleration (k2z). The function returns 
    the new position (new_z) as z+k2z*h.
    
    Arguments
        f(t,z,...)
            function that contains the RHS of the equation dz/dt = f(t,z,...)
    
        t
            the time at which the system is currently at. Only impacts anything
            if f() has a time dependence
        
        h
            the timestep between each run of the method, smaller timesteps will
            result in more accurate results, but increase computation time
    
        args (tuple, optional)
            additional arguments to pass to f
    
    Returns
        znew = z(t+h)
    """

    if not isinstance(args,tuple):
        args = (args,)
    
    # detemines initial acceleration
    k1z = f(t, z, *args)
            
    # finds new acc from k1
    k2z = f(t+h/2, z+h/2*k1z, *args)
            
    # changes r/v in next time step based on val from k2
    # moves k2 forward a timestep when using
    znew = z+k2z*h
    return znew

def rk4(f,t,z,h,args=()):
    """
    This sets up the 4th-order Runge-Kutta by defining it as a function which 
    takes in a function (f), time (t), position (z), time-step (h), and other 
    arguments (args). It finds old and new acceleration values four times, 
    before returning the new position (new_z) as z+k4z*h.
    
    Arguments
        f(t,z,...)
            function that contains the RHS of the equation dz/dt = f(t,z,...)
    
        t
            the time at which the system is currently at. Only impacts anything
            if f() has a time dependence
        
        h
            the timestep between each run of the method, smaller timesteps will
            result in more accurate results, but increase computation time
    
        args (tuple, optional)
            additional arguments to pass to f
    
    Returns
        znew = z(t+h)
    """
   
    if not isinstance(args,tuple):
        args = (args,)
    # detemines initial acc
    k1z =  f(t,z,*args) 
    
    # # finds new acc from k1
    k2z =  f(t+h/2,z+h/2*k1z,*args)
    
    # # finds new acc from k2
    k3z =  f(t+h/2,z+h/2*k2z,*args)
    
    # # finds new acc from k3
    k4z = f(t+h,z+h*k3z,*args)
    
    # # changes r/v in next time step based on values from the k's
    znew = z + (h/6)*(k1z + 2*k2z + 2*k3z + k4z)
    return znew