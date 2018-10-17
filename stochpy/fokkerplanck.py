"""
Numerical solver for the Fokker-Planck equations
"""
import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sps
from . import edpy


class FokkerPlanck1D(object):
    """
    Solver for the 1D Fokker-Planck equation

    d_t P(x,t) = - a(x,t)d_x P(x,t) + D d^2 P(x,t)/d_x^2

    This is just the legacy code which was migrated from the StochModel1D class.
    It should be rewritten with a better structure.
    In particular, it only works with a constant diffusion for now.
    """
    def __init__(self, drift, diffusion):
        """
        drift: function of two variables (x, t)
        diffusion: diffusion coefficient (float).
        """
        self.drift = drift
        self.diffusion = diffusion

    def _fpeq(self, P, X, t):
        """ Right hand side of the Fokker-Planck equation """
        return -X.grad(self.drift(X.grid, t)*P) + self.diffusion*X.laplacian(P)

    def _fpadj(self, G, X, t):
        """
        The adjoint of the Fokker-Planck operator, useful for instance
        in first passage time problems for homogeneous processes.
        """
        return self.drift(X.grid, t)[1:-1]*X.grad(G)+self.diffusion*X.laplacian(G)

    def _fpmat(self, X, t):
        """
        Sparse matrix representation of the linear operator
        corresponding to the RHS of the FP equation
        """
        return -X.grad_mat()*sps.dia_matrix((self.drift(X.grid, t), np.array([0])),
                                            shape=(X.N, X.N)) + self.diffusion*X.lapl_mat()

    def _fpadjmat(self, X, t):
        """ Sparse matrix representation of the adjoint of the FP operator """
        return sps.dia_matrix((self.drift(X.grid, t)[1:-1], np.array([0])),
                              shape=(X.N-2, X.N-2))*X.grad_mat() + self.diffusion*X.lapl_mat()

    def _fpbc(self, fdgrid, bc=('absorbing', 'absorbing')):
        """ Build the boundary conditions for the Fokker-Planck equation and return it.
        This is useful when at least one of the sides is a reflecting wall. """
        dx = fdgrid.dx
        dic = {('absorbing', 'absorbing'): edpy.DirichletBC([0, 0]),
               ('absorbing', 'reflecting'): edpy.BoundaryCondition(lambda Y, X, t: [0,Y[-2]/(1-self.drift(X[-1], t)*dx/self.diffusion)]),
               ('reflecting', 'absorbing'): edpy.BoundaryCondition(lambda Y, X, t: [Y[1]/(1+self.drift(X[0], t)*dx/self.diffusion),0]),
               ('reflecting', 'reflecting'): edpy.BoundaryCondition(lambda Y, X, t: [Y[1]/(1+self.drift(X[0], t)*dx/self.diffusion), Y[-2]/(1-self.drift(X[-1], t)*dx/self.diffusion)])}
        if bc not in dic:
            raise NotImplementedError("Unknown boundary conditions for the Fokker-Planck equations")
        return edpy.DirichletBC([0, 0]) if self.diffusion == 0 else dic[bc]

    def fpintegrate(self, t0, T, **kwargs):
        """
        Numerical integration of the associated Fokker-Planck equation, or its adjoint.
        Optional arguments are the following:
        - bounds=(-10.0,10.0); domain where we should solve the equation
        - npts=100;            number of discretization points in the domain (i.e. spatial resolution)
        - dt;                  timestep (default choice suitable for the heat equation with forward scheme)
        - bc;                  boundary conditions (either a BoundaryCondition object or a tuple sent to _fpbc)
        - method=euler;        numerical scheme: explicit (default), implicit, or crank-nicolson
        - adj=False;           integrate the adjoint FP rather than the forward FP?
        """
        # Get computational parameters:
        B, A = kwargs.pop('bounds', (-10.0, 10.0))
        Np = kwargs.pop('npts', 100)
        fdgrid = edpy.RegularCenteredFD(B, A, Np)
        dt = kwargs.pop('dt', 0.25*(np.abs(B-A)/(Np-1))**2/self.diffusion)
        bc = self._fpbc(fdgrid, **kwargs)
        method = kwargs.pop('method', 'euler')
        adj = kwargs.pop('adjoint', False)
        # Prepare initial P(x):
        P0 = kwargs.pop('P0', 'gauss')
        if P0 == 'gauss':
            P0 = np.exp(-0.5*((fdgrid.grid-kwargs.get('P0center', 0.0))/kwargs.get('P0std', 1.0))**2)/(np.sqrt(2*np.pi)*kwargs.get('P0std', 1.0))
            P0 /= integrate.trapz(P0, fdgrid.grid)
        if P0 == 'dirac':
            P0 = np.zeros_like(fdgrid.grid)
            np.put(P0, len(fdgrid.grid[fdgrid.grid < kwargs.get('P0center', 0.0)]), 1.0)
            P0 /= integrate.trapz(P0, fdgrid.grid)
        if P0 == 'uniform':
            P0 = np.ones_like(fdgrid.grid)
            P0 /= integrate.trapz(P0, fdgrid.grid)
        # Numerical integration:
        if T > 0:
            if method in ('impl', 'implicit', 'bwd', 'backward',
                          'cn', 'cranknicolson', 'crank-nicolson'):
                fpmat = {False: self._fpmat, True: self._fpadjmat}.get(adj)
                return edpy.EDPLinSolver().edp_int(fpmat, fdgrid, P0, t0, T, dt, bc, scheme=method)
            else:
                fpfun = {False: self._fpeq, True: self._fpadj}.get(adj)
                return edpy.EDPSolver().edp_int(fpfun, fdgrid, P0, t0, T, dt, bc)
        else:
            return t0, fdgrid.grid, P0

    def pdfgen(self, *args, **kwargs):
        """ Generate the pdf solution of the FP equation at various times """
        t0 = kwargs.pop('t0', args[0])
        fun = kwargs.pop('integ', self.fpintegrate)
        for t in args:
            t, X, P = fun(t0, t-t0, **kwargs)
            t0 = t
            kwargs['P0'] = P
            yield t, X, P
