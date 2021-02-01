# -*- coding: utf-8 -*-

#from _future_ import division, absolute_import, print_function
from ..finitediff import jacobian as _jacobian, hessian as _hessian
from ..linesearch import xstep as _xstep, backtracking as _backtracking, interp23 as _interp23, unimodality as _unimodality, golden_section as _golden_section
from .. import xp as _xp
from .. import sl as _sl


class unconstrained(object):

    def __init__(self,parameters):
        self.eps = _xp.finfo(_xp.float64).eps
        self.resolution = _xp.finfo(_xp.float64).resolution
        self.params = parameters
        self._ls_algorithms = {'backtracking':_backtracking,
                'interp23':_interp23,
                'unimodality':_unimodality,
                'golden-section':_golden_section}
        self._unc_algorithms = {'gradient':self._gradient,
                'newton':self._newton,
                'modified-newton':self._mod_newton,
                'conjugate-gradient':self._conj_gradient,
                'fletcher-reeves':self._fletcher_reeves,
                'quasi-newton':self._qn}
        #-----------------------------------------------------------------------
        #Init params
        self.d = None; self.g = None; self.Q = None
        self.x = None; self.x_vec = None
        self.fun = None; self.jac = None
        #Linesearch params
        self.ls = None
        #Fmin params
        self.fmin_method = None
        self.count = 0 #count # loops

    def _gradient(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            gradient step (linear convergence)
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        x0 = _xp.array(x0) #ensure proper cast
        g = _jacobian(fun,x0,**self.params['jacobian'])
        return -g, g, []

    def _newton(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            newton method (quadratic convergence)
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        x0 = _xp.array(x0) #ensure proper cast
        g = _jacobian(fun,x0,**self.params['jacobian'])
        Q = _hessian(fun,x0,**self.params['hessian'])
        return -_xp.linalg.inv(Q).dot(g), g, Q

    def _mod_newton(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            newton method
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        x0 = _xp.array(x0) #ensure proper cast
        g = _jacobian(fun,x0,**self.params['jacobian'])
        Q = _hessian(fun,x0,**self.params['hessian'])
        eigs, nu = _xp.linalg.eig(Q)
        eigs = abs(eigs)
        eigs[eigs<self.params['fminunc']['params']['modified-newton']['sigma']] = self.params['fminunc']['params']['modified-newton']['sigma']
        d = _xp.array(_sl.cho_solve(_sl.cho_factor(nu.dot(_xp.diag(_xp.array(eigs))).dot(nu.T)),-g))
        return d, g, Q

    def _conj_gradient(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            conjugate-gradient method
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        x0 = _xp.array(x0) #ensure proper cast
        d0 = _xp.array(d0)
        g = _jacobian(fun,x0,**self.params['jacobian'])
        Q = _hessian(fun,x0,**self.params['hessian'])
        if sum(abs(d0)) < self.eps or ((kwargs['iters'] + 1) % len(x0)) < self.eps:
            return -g, g, Q
        else:
            beta = g.T.dot(Q).dot(d0)/(d0.T.dot(Q).dot(d0))
            d = -g + beta*d0
            return d, g, Q

    def _fletcher_reeves(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            fletcher_reeves
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        x0 = _xp.array(x0) #ensure proper cast
        g0 = _xp.array(g0)
        d0 = _xp.array(d0)
        g = _jacobian(fun,x0,**self.params['jacobian'])
        if sum(abs(d0)) < self.eps or ((kwargs['iters'] + 1) % len(x0)) < self.eps:
            return -g, g, []
        else:
            beta = g.T.dot(g)/(g0.T.dot(g0))
            d = -g + beta*d0
            return d, g, []

    def _dfp(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            Davidon-Flether-Powell
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        x0 = _xp.array(x0) #ensure proper cast
        g0 = _xp.array(g0)
        d0 = _xp.array(d0)
        Q0 = _xp.array(Q0)
        g = _jacobian(fun,x0,**self.params['jacobian'])
        if sum(abs(d0)) < self.eps or ((kwargs['iters'] + 1) % len(x0)) < self.eps:
            Q = _xp.array([self.params['hessian']['initial'] if self.params['hessian']['initial'] else _xp.identity(len(x0))][0])
        else:
            q = (g-g0)[_xp.newaxis].T; q = _xp.array(q)
            p = (kwargs['alpha']*d0)[_xp.newaxis].T; p = _xp.array(p)
            Q = Q0 + _xp.dot(p,p.T)/_xp.dot(p.T,q) - ( _xp.dot(Q0,q).dot(_xp.dot(q.T,Q0)))/(_xp.dot(q.T,Q0).dot(q))
        d = -Q.dot(g)
        return d, g, Q

    def _bfgs(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            Broyden-Fletcher-Goldfarb-Shanno
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        print("Start BFGS")
        x0 = _xp.array(x0) #ensure proper cast
        g0 = _xp.array(g0)
        d0 = _xp.array(d0)
        Q0 = _xp.array(Q0)
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else: # only for init
            alpha = 1
        if kwargs['J'] != None: #use analytic jacobian
            g = kwargs['J'](x0)
        else: #use finite differences
            g = _jacobian(fun,x0,**self.params['jacobian'])
        # if sum(abs(d0)) < self.eps or ((kwargs['iters'] + 1) % len(x0)) < self.eps: #if stepsize was effectively zero, or is even iter #
        if sum(abs(alpha*d0)) < self.eps: #if stepsize was effectively zero
            #If set Q=Id, then step equivalent to gradient descent
            Q = _xp.array([self.params['hessian']['initial'] if self.params['hessian']['initial'] else _xp.identity(len(x0))][0])
            # Q = _xp.array([_xp.identity(len(x0))][0])
        else:
            #Sherman-Morrison formula for estimating inverse Hessian
            q = (g-g0)[_xp.newaxis].T #finite difference of jac
            p = _xp.array((alpha*d0)[_xp.newaxis].T) #x step dif
            print("Jac Finite Dif:{}".format(str(q)))
            print("X Finite Dif:{}".format(str(p)))
            Q = Q0 + (1.0 + q.T.dot(Q0).dot(q)/(q.T.dot(p)))*(p.dot(p.T))/(p.T.dot(q)) - (p.dot(q.T).dot(Q0)+Q0.dot(q).dot(p.T))/(q.T.dot(p))
        d = -Q.dot(g) #compute direction that solves H*d = -J, where H is hessian, J is Jacobian
        print("x0:{}".format(str(x0)))
        print("Jacobian:{}".format(str(g)))
        print("Inverse Hessian:{}".format(str(Q)))
        print("Descent direction:{}".format(str(d)))
        return d, g, Q

    def _qn(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            Quasi-Newton caller
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        x0 = _xp.array(x0) #ensure proper cast
        g0 = _xp.array(g0)
        d0 = _xp.array(d0)
        Q0 = _xp.array(Q0)
        if self.params['fminunc']['params']['quasi-newton']['hessian_update'] in ('davidon-fletcher-powell','dfp'):
            return self._dfp(fun,x0,d0,g0,Q0,*args,**kwargs)
        elif self.params['fminunc']['params']['quasi-newton']['hessian_update'] in ('broyden-fletcher-goldfarb-shanno','BFGS','bfgs'):
            return self._bfgs(fun,x0,d0,g0,Q0,*args,**kwargs)
        else:
            raise Exception('Hessian update method ({}) not implemented'.format(self.params['fminunc']['params']['quasi-newton']['hessian_update']))

    def _update(self, vectorized):
        ls_alg = self._ls_algorithms[self.params['linesearch']['method']]
        ls_kwargs = self.params['linesearch']['params'][self.params['linesearch']['method']]
        self.ls = ls_alg(self.fun,self.x,self.d,**ls_kwargs, J=self.jac)
        alpha = self.ls['alpha']
        lsiters += self.ls['iterations']
        #
        self.x = self._xstep(self.x,self.d,alpha)
        if vectorized:
            self.x_vec += [self.x]
        else:
            pass
        self.d, self.g, self.Q = self.fmin_method(self.fun,self.x,self.d,self.g,self.Q, \
                                                  iters=iters,alpha=alpha,J=self.jac) #update d, g, Q

    def fminunc(self,fun,x0,threshold=1e-6,vectorized=False,**kwargs):
        '''
            Minimum Unconstrained Optimization
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
            ..threshold as a numeric value; threshold at which to stop the iterations
            ..**kwargs = initial_hessian : as matrix (default = identity)
            .. see unconstrained.params for further details on the methods that are being used
        '''
        #-----------------------------------------------------------------------
        #Set up minimization parameters
        self.fun = fun
        if 'J' in kwargs: #use analytic Jacobian
            self.jac=kwargs['J']
        if 'max_iter' in kwargs:
            max_iter= kwargs['max_iter']
        else:
            max_iter = self.params['fminunc']['params'][self.params['fminunc']['method']]['max_iter']
        #-----------------------------------------------------------------------
        #Minimization
        if self.count == 0:
            ##Initial Gradient Descent step
            print("Initial GD step")
            self.fmin_method = self._unc_algorithms[self.params['fminunc']['method']]
            self.d, self.g, self.Q = self.fmin_method(self.fun,x0,_xp.zeros(len(x0)),\
                                                      [],[],iters=0,J=self.jac)
            if vectorized:
                self.x_vec = [x0]
            else:
                pass
        ##Loop
        self.x = _xp.array(x0)
        iters = 0
        lsiters = 0
        while _xp.dot(self.g,self.g) > threshold and iters < max_iter:
            print("fmin iteration: {}".format(iters), end='\r')
            self._update(vectorized)
            iters += 1
        print("fmin max iter: {} out of {}".format(iters, max_iter))
        self.count += 1
        if vectorized:
            return {'x':self.x_vec, 'f':[self.fun(x) for x in self.x_vec], 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}
        else:
            return {'x':self.x, 'f':self.fun(self.x), 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}
