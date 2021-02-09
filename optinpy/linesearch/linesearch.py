# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function
from .. import xp as __xp
from ..finitediff import jacobian as __jacobian, hessian as __hessian

eps = __xp.finfo(__xp.float64).eps

def xstep(x0,d,alpha):
    '''
        returns x_1 given x_0, d and alpha --> x1 = x0+(alpha*d)
    '''
    return __xp.array(x0,__xp.float64)+alpha*__xp.array(d,__xp.float64)

def __armijo(fun,x0,d,dd,alpha,c):
    '''
        Check's whether Armijo's conditions upholds
        i.e. fun(x0+alpha*d) <= fun(x0) + c*alpha*dd
    '''
    return fun(xstep(x0,d,alpha)) <= fun(x0)+c*alpha*dd

def backtracking(fun,x0,d=None,alpha=1,rho=0.6,c=1e-4,max_iter=1e3,J=None,**kwargs):
    '''
        backtracking algorithm, arg_min(alpha) fun(x0+alpha*d)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point at which the __jacobian should be estimated
        ..d as a numeric array: direction towards which to walk
        ..alpha as a numeric value; maximum step from x_i
        ..rho as a numeric value; rate of decrement in alpha
        ..c as a numeric values; constant of Wolfe's condition; defaults to Nocedal's example (1e-4)
        ..**kwargs as a dictionary with the jacobian function parameters
    '''
    print("LS Method: Backtracking")
    d = __xp.array(d) #ensure proper cast
    #Compute Jacobian
    if J != None:
        g = kwargs['J'](x0) #use analytic Jacobian
    else:
        g = __jacobian(fun,x0,**kwargs)
    if any(d): #check if d contains elements other than 0
        pass
        dd = __xp.dot(d,g)
    else: #if d is all zeros
        d = -__xp.array(g,__xp.float64) # steepest descent step
        dd = __xp.dot(d,-d)
    #Backtracking linesearch
    iters = 0
    dx = __xp.inf #init
    dx_tol = 1e-4
    print('dd:{}'.format(str(dd)))
    while not __armijo(fun,x0,d,dd,alpha,c) and iters < max_iter and __xp.sum(__xp.abs(dx))>dx_tol:# Armijo's Condition
        print("ls iteration: {}".format(iters), end='\r')
        alpha = rho*alpha #take successively smaller steps along direction d
        dx=xstep(x0,d,alpha)
        iters += 1
    print("dx:{}".format(str(c*alpha*dd)))
    print("ls max iter: {}".format(iters))
    return {'x':xstep(x0,d,alpha), 'f':fun(xstep(x0,d,alpha)), 'alpha':alpha, 'iterations':iters}

def interp23(fun,x0,d=None,alpha=1,c=1e-4,alpha_min=0.1,rho=0.5,max_iter=1e3,J=None,**kwargs):
    '''
        interpolating algorithm, arg_min(alpha) fun(x0+alpha*d)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point at which the __jacobian should be estimated
        ..d as a numeric array: direction towards which to walk
        ..alpha as a numeric value; maximum step from x_i
        ..c as a numeric values; constant of Armijo's condition
        ..alpha_min; lower limit for alpha when alpha is too small
        ..**kwargs as a dictionary with the jacobian function parameters
    '''
    print("LS Method: Interp23")
    alpha0 = alpha
    f0 = fun(x0)
    if J != None:
        g = kwargs['J'](x0) #use analytic Jacobian
    else:
        g = __jacobian(fun,x0,**kwargs)
    if any(d):
        dd = __xp.dot(d,g)
    else:
        d = -__xp.array(g,__xp.float64) # steepest descent step
        dd = __xp.dot(d,-d)
    iters = {'first_order':0,'second_order':0,'third_order':0}
    iters['first_order'] += 1
    if __armijo(fun,x0,d,dd,alpha0,c):
        print("First-order")
        print("new loss:{}".format(fun(xstep(x0,d,alpha0))))
        print("Armijo:{}".format(fun(x0)+c*alpha0*dd))
        return {'x':xstep(x0,d,alpha0), 'f':fun(xstep(x0,d,alpha0)), 'alpha':alpha0, 'iterations':sum(iters.values()), 'inner_iterations':iters}
    else: #Quadratic Interpolation
        # second order approximation
        iters['second_order'] += 1
        alpha1 = -(dd*alpha0**2)/(2*(fun(xstep(x0,d,alpha0))-f0-dd*alpha0))
        if __armijo(fun,x0,d,dd,alpha1,c) and alpha1 > alpha_min:# Armijo's Condition
            print("Second-order")
            print("new loss:{}".format(fun(xstep(x0,d,alpha1))))
            print("Armijo:{}".format(fun(x0)+c*alpha1*dd))
            return {'x':xstep(x0,d,alpha1), 'f':fun(xstep(x0,d,alpha1)), 'alpha':alpha1, 'iterations':sum(iters.values()), 'inner_iterations':iters}
        else:
            alpha1 = alpha0*rho
            iters['second_order'] += 1
            if __armijo(fun,x0,d,dd,alpha1,c) and alpha1 > alpha_min: # check whether a single backtracking iteration gives rise to a reasonable stepsize
                print("Second-order")
                print("new loss:{}".format(fun(xstep(x0,d,alpha1))))
                print("Armijo:{}".format(fun(x0)+c*alpha1*dd))
                return {'x':xstep(x0,d,alpha1), 'f':fun(xstep(x0,d,alpha1)), 'alpha':alpha1,'iterations':sum(iters.values()), 'inner_iterations':iters}
            else:
                while not __armijo(fun,x0,d,dd,alpha1,c) and iters['third_order'] < max_iter: #Cubic Interpolation
                    iters['third_order'] += 1
                    coeff = (1/(alpha0**2*alpha1**2*(alpha1-alpha0)))
                    m = __xp.array([[alpha0**2,-alpha1**2],[-alpha0**3,alpha1**3]])
                    v = __xp.array([fun(xstep(x0,d,alpha1))-f0-alpha1*dd,fun(xstep(x0,d,alpha0))-f0-alpha0*dd])
                    a, b = coeff*__xp.dot(m,v)
                    alpha0 = float(alpha1)
                    alpha1 = float((-b+(b**2.-3.*a*dd)**0.5)/(3.*a))
            print("ls max iter: {}".format(sum(iters.values())))
            print("new loss:{}".format(fun(xstep(x0,d,alpha1))))
            print("Armijo:{}".format(fun(x0)+c*alpha1*dd))
            return {'x':xstep(x0,d,alpha1), 'f':fun(xstep(x0,d,alpha1)), 'alpha':alpha1, 'iterations':sum(iters.values()), 'inner_iterations':iters}

def unimodality(fun,x0,d=None,b=1,threshold=0.01,max_iter=1e3,**kwargs):
    '''
        unimodality algorithm, arg_min(alpha) fun(x0+alpha*d)
        ..fun, a callable object, be strictly quasiconvex between the interval [a,b], a0 = 0
        ..x0 as a numeric array; starting point
        ..d as a numeric array: direction towards which to walk
        ..b as the upper bound for the interval
        ..threshold min mean-relative difference between interval bounds as of which the procedeure ceases flowing
    '''
    interv = __xp.array([0,b])
    if any(d):
        pass
    else:
        d = -__xp.array(__jacobian(fun,x0,**kwargs),__xp.float64) # steepest descent step
    iters = 0
    while 2*abs((interv[-1]-interv[0])/(abs(interv[-1])+abs(interv[0]))) > threshold and iters < max_iter:
        iters += 1
        x = __xp.sort(__xp.random.uniform(*interv+[2])) # evaluate alpha and beta uniformily distributed in the interv
        phi = [fun(xstep(x0,d,x_)) for x_ in x]
        if phi[0] > phi[1]:
            interv[0] = x[0]
        else:
            interv[1] = x[1]
    alpha = (interv[1]+interv[0])/2.0
    return {'x':xstep(x0,d,alpha), 'f':fun(xstep(x0,d,alpha)), 'alpha':alpha, 'iterations':iters}

def golden_section(fun,x0,d=None,b=1,threshold=0.01,max_iter=1e3,**kwargs):
    '''
        golden-section algorithm, arg_min(alpha) fun(x0+alpha*d)
        ..fun, a callable object (function)
        ..x0 as a numeric array; starting point
        ..d as a numeric array: direction towards which to walk
        ..b as the upper bound for the alpha domain in Re (b>0)
        ..threshold min mean-relative difference between interval bounds as of which the procedeure ceases flowing
    '''
    if any(d):
        pass
    else:
        d = -__xp.array(__jacobian(fun,x0,**kwargs),__xp.float64) # steepest descent step
    r = (5**.5-1)/2.0
    interv = [0,b]
    alpha = interv[0] + (1-r)*b
    beta = interv[0] + r*b
    phi = [fun(xstep(x0,d,x_)) for x_ in [alpha,beta]]
    iters = 0
    while 2*abs((interv[-1]-interv[0])/(abs(interv[-1])+abs(interv[0]))) > threshold  and iters < max_iter:
        iters += 1
        if phi[0] > phi[1]:
            interv[0] = alpha
            alpha = beta
            phi[0] = phi[1]
            beta = alpha + r*(interv[1]-interv[0])
            phi[1] = fun(xstep(x0,d,beta))
        else:
            interv[1] = beta
            beta = alpha
            phi[1] = phi[0]
            alpha = interv[0] + (1-r)*(interv[1]-interv[0])
            phi[0] = fun(xstep(x0,d,alpha))
    alpha = (interv[1]+interv[0])/2.0
    return {'x':xstep(x0,d,alpha), 'f':fun(xstep(x0,d,alpha)), 'alpha':alpha, 'iterations':iters}
