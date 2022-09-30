import numpy as np
import matplotlib.pyplot as plt

"""
Basically ripped most of this from PHY 415, except now
it's object-oriented and now way more confusing than it needs to be,
but if I actually got it to work, it should run for any ODE... so long
as there's no time-dependence in the equation... I gotta add that soon
"""

class init:
    """
    A function that I use to initialize all the values for the
    iterative functions. Kinda extra, but it helps save some space.
    
    args:
        
        ndim: number of dimensions we are working in
        
        tfinal: how long to iterate our method for
        
        N: number of iterations
        
        r0: initial position. If in higher dimensions,
        include as a list of values for the x,y,z... positions
        
        v0: initial velocities. See above for what to do for
        higher dimensions
        
    returns:
        
        dt: time between each iteration
        
        t: time array
        
        v: empty velocity array
        
        r: empty position array
    """
    def __init__(self, ndim, tfinal, N, r0, v0):
        
        self.dt = tfinal/N
        self.t = np.arange(0,tfinal,self.dt)
        
        self.v = np.zeros([N,ndim])
        self.r = np.zeros([N,ndim])
        
        self.v[0] = v0
        self.r[0] = r0        

class euler(init):
    """
    class for running the Euler method
    
    args:
        
        ndim: number of dimensions we are working in
        
        tfinal: how long to iterate our method for
        
        N: number of iterations
        
        r0: initial position. If in higher dimensions,
        include as a list of values for the x,y,z... positions
        
        v0: initial velocities. See above for what to do for
        higher dimensions
    
    methods:
        
        iterate():
            where the magic happens. Runs the method Euler method and 
            generates position, velocity and time arrays
            
            args:
                a_func: the equation that we're iterating through
                should take in a position and velocity, reguardless 
                of whether or not it actually uses it
            
            returns:
                r: position array
                
                v: velocity array
                
                t: time array
    """
    def __init__(self, ndim, tfinal, N, r0, v0):
        
        # initializes variables. see init func for specifics
        super().__init__(ndim, tfinal, N, r0, v0)
    
    def iterate(self,a_func):
        
        for i in range(len(self.t)-1):
            # determines "acceleration" of the object 
            a =  a_func(self.r[i], self.v[i])
            
            # changes r/v in next time step depending on acceleration
            self.r[i+1], self.v[i+1] = self.r[i] + self.dt*self.v[i], self.v[i] + self.dt*a[1]
        
        return self.r,self.v, self.t
    
class rk2(init):
    """
    class for running the Runge Kutta 2nd order method
    
    args:
        
        ndim: number of dimensions we are working in
        
        tfinal: how long to iterate our method for
        
        N: number of iterations
        
        r0: initial position. If in higher dimensions,
        include as a list of values for the x,y,z... positions
        
        v0: initial velocities. See above for what to do for
        higher dimensions
    
    methods:
        
        iterate():
            where the magic happens. Runs the method Runge Kutta (2nd order) 
            method and generates position, velocity and time arrays
            
            args:
                a_func: the equation that we're iterating through
                should take in a position and velocity, reguardless 
                of whether or not it actually uses it
            
            returns:
                r: position array
                
                v: velocity array
                
                t: time array
    """    
    def __init__(self, ndim, tfinal, N, r0, v0):
        
        # initializes variables. see init func for specifics
        super().__init__(ndim, tfinal, N, r0, v0)
        
    def iterate(self, a_func):
        
        for i in range(len(self.t)-1):
            
            # detemines initial acceleration
            k1r, k1v =  a_func(self.r[i], self.v[i])
            
            # moves acc forwards by 1 timestep
            k1r, k1v = k1r*self.dt, k1v*self.dt
            
            # finds new acc from k1
            k2r, k2v = a_func(self.r[i]+0.5*k1r, self.v[i]+0.5*k1v)
            
            # changes r/v in next time step based on val from k2
            # also moves k2 forward a timestep when using
            self.r[i+1], self.v[i+1] = self.r[i]+k2r*self.dt, self.v[i]+k2v*self.dt,
            
        return self.r, self.v, self.t
    
class rk4(init):
    """
    class for running the Runge Kutta 4th order method
    
    args:
        
        ndim: number of dimensions we are working in
        
        tfinal: how long to iterate our method for
        
        N: number of iterations
        
        r0: initial position. If in higher dimensions,
        include as a list of values for the x,y,z... positions
        
        v0: initial velocities. See above for what to do for
        higher dimensions
    
    methods:
        
        iterate():
            where the magic happens. Runs the method Runge Kutta (4th order) 
            method and generates position, velocity and time arrays
            
            args:
                a_func: the equation that we're iterating through
                should take in a position and velocity, reguardless 
                of whether or not it actually uses it
            
            returns:
                r: position array
                
                v: velocity array
                
                t: time array
    """    
    def __init__(self, ndim, tfinal, N, r0, v0):
        
        # initializes variables. see init func for specifics
        super().__init__(ndim, tfinal, N, r0, v0)
    
    def iterate(self, a_func):
        
        for i in range(len(self.t)-1):
            
            # detemines initial acc, moves it forward by 1 timestep
            k1r, k1v =  a_func(self.r[i], self.v[i])
            k1r, k1v = k1r*self.dt, k1v*self.dt 
            
            # finds new acc from k1, moves it forward by 1 timestep
            k2r, k2v =  a_func(self.r[i]+0.5*k1r, self.v[i]+0.5*k1v)
            k2r, k2v = k2r*self.dt, k2v*self.dt 
            
            # finds new acc from k2, moves it forward by 1 timestep
            k3r, k3v =  a_func(self.r[i]+0.5*k2r, self.v[i]+0.5*k2v)
            k3r, k3v = k3r*self.dt, k3v*self.dt 
            
            # finds new acc from k3, moves it forward by 1 timestep in final line
            k4r, k4v = a_func(self.r[i]+0.5*k3r, self.v[i]+0.5*k3v)
            
            # changes r/v in next time step based on val from k4
            self.r[i+1], self.v[i+1] = self.r[i]+k4r*self.dt, self.v[i]+k4v*self.dt,
            
        return self.r, self.v, self.t
    

    
    
    
    
    
    
