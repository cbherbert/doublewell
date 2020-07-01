"""
Simulating diffusion processes in arbitrary dimensions
=================================================================

.. currentmodule:: stochrare.dynamics.diffusion

This module defines the `DiffusionProcess` class, representing generic diffusion processes with
arbitrary drift and diffusion coefficients, in arbitrary dimension.

This class can be subclassed for specific diffusion processes for which methods can be specialized,
both to simplify the code (e.g. directly enter analytical formulae when they are available) and for
performance.
As an exemple of this mechanism, we also provide in this module the `ConstantDiffusionProcess`
class, for which the diffusion term is constant and proportional to the identity matrix,
the `OrnsteinUhlenbeck` class representing the particular case of the Ornstein-Uhlenbeck process,
and the `Wiener` class corresponding to Brownian motion.
These classes form a hierarchy deriving from the base class, `DiffusionProcess`.

.. autoclass:: DiffusionProcess
   :members:

.. autoclass:: ConstantDiffusionProcess
   :members:

.. autoclass:: OrnsteinUhlenbeck
   :members:

.. autoclass:: Wiener
   :members:

"""
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.misc import derivative
from numba import jit
from ..utils import pseudorand, method1d

class DiffusionProcess:
    r"""
    Generic class for diffusion processes in arbitrary dimensions.

    It corresponds to the family of SDEs :math:`dx_t = F(x_t, t)dt + \sigma(x_t, t)dW_t`,
    where :math:`F` is a time-dependent :math:`N`-dimensional vector field
    and :math:`W` the :math:`M`-dimensional Wiener process.
    The diffusion matrix sigma has size NxM.

    Attributes
    ----------
    drift : function with two arguments
        The vector field :math:`F(x, t)`.
    diffusion : function with two arguments
        The diffusion coefficient :math:`\sigma(x, t)`.
    dimension : int
        The dimension of the process.
    """

    default_dt = 0.1

    def __init__(self, vecfield, sigma, dimension, **kwargs):
        """
        vecfield: vector field
        sigma: diffusion coefficient (noise)
        dimension: int

        vecfield and sigma are functions of two variables (x,t).
        """
        self._drift = jit(vecfield, nopython=True)
        self._diffusion = jit(sigma, nopython=True)
        self.dimension = dimension
        self.__deterministic__ = kwargs.get('deterministic', False)


    @property
    def drift(self):
        return self._drift

    @drift.setter
    def drift(self, driftnew):
        self._drift = jit(driftnew, nopython=True)

    @property
    def diffusion(self):
        return self._diffusion

    @diffusion.setter
    def diffusion(self, diffusionnew):
        self._diffusion = jit(diffusionnew, nopython=True)

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, dimensionnew):
        if dimensionnew < 1:
            raise ValueError("Attribute dimension cannot be lower than 1")
        self._dimension = dimensionnew


    @method1d
    def potential(self, X, t):
        """
        Compute the potential from which the force derives.

        Parameters
        ----------
        X : ndarray (1d)
            The points where we want to compute the potential.

        Returns
        -------
        V : ndarray (1d)
            The potential from which the force derives, at the given points.

        Notes
        -----
        We integrate the vector field to obtain the value of the underlying potential
        at the input points.
        Caveat: This works only for 1D dynamics.
        """
        if self.dimension != 1:
            raise ValueError('Generic dynamics in arbitrary dimensions are not gradient dynamics!')
        fun = interp1d(X, -1*self.drift(X, t), fill_value='extrapolate')
        return np.array([integrate.quad(fun, 0.0, x)[0] for x in X])


    def update(self, xn, tn, **kwargs):
        r"""
        Return the next sample for the time-discretized process.

        Parameters
        ----------
        xn: ndarray
            A n-dimensional vector (in :math:`\mathbb{R}^n`).
        tn: float
            The current time.

        Keyword Arguments
        ------------------
        dt : float
            The time step.
        dw : ndarray
            The brownian increment if precomputed.
            By default, it is generated on the fly from a Gaussian
            distribution with variance :math:`dt`.

        Returns
        -------
        x : ndarray
            The position at time tn+dt.

        Notes
        -----
        This method uses the Euler-Maruyama method [1]_ [2]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n`,
        for a fixed time step :math:`\Delta t`, where :math:`\Delta W_n` is a random vector
        distributed according to the standard normal distribution [1]_ [2]_.

        It is the straightforward generalization to SDEs of the Euler method for ODEs.

        The Euler-Maruyama method has strong order 0.5 and weak order 1.

        References
        ----------
        .. [1] G. Maruyama, "Continuous Markov processes and stochastic equations",
           Rend. Circ. Mat. Palermo 4, 48-90 (1955).
        .. [2] P. E. Kloeden and E. Platen,
           "Numerical solution of stochastic differential equations", Springer (1992).
        """
        dt = kwargs.get('dt', self.default_dt)
        dim = self.dimension
        if dim > 1:
            dw = kwargs.get('dw', np.random.normal(0.0, np.sqrt(dt), dim))
            return xn + self.drift(xn, tn)*dt+self.diffusion(xn, tn) @ dw
        else:
            dw = kwargs.get('dw', np.random.normal(0.0, np.sqrt(dt)))
            return xn + self.drift(xn, tn)*dt+self.diffusion(xn, tn) * dw


    def _integrate_brownian_path(self, dw, num, ratio):
        """
        Return piece-wise integrated brownian path.

        Parameters
        ----------
        dw: ndarray
          Brownian path.
        num: int
          Number of SDE timesteps.
        dim: int
          Brownian path dimension.
        ratio: int
          Ratio between brownian path timestep and SDE timestep.

        Returns
        -------
        integrated_dw: ndarray
          Piecewise integrated brownian path.
        """

        expected_shape = ((num-1)*ratio, self.dimension) if self.dimension > 1 else ((num-1)*ratio,)
        if not dw.shape == expected_shape:
            raise ValueError("Brownian path array has dimension {}, expected {}".format(dw.shape, expected_shape))
        if self.dimension > 1:
            integrated_dw = np.zeros((num-1, self.dimension), dtype=dw.dtype)
            for coord in range(self.dimension):
                integrated_dw[:,coord] = dw[:,coord].reshape((num-1, ratio)).sum(axis=1)
        else:
            integrated_dw = dw = dw.reshape((num-1, ratio)).sum(axis=1)

        return integrated_dw

    def integrate_sde(self, x, t, w, **kwargs):
        r"""
        Dispatch SDE integration for different numerical schemes

        Parameters
        ----------
        x: ndarray
            The (empty) position array
        t: ndarray
            The sample time array
        w: ndarray
            The brownian motion realization used for integration

        Keyword Arguments
        -----------------
        method: str
            The numerical scheme: 'euler' (default) or 'milstein'
        dt: float
            The time step

        Notes
        -----
        We define this method rather than putting the code in the `trajectory` method to make
        it easier to implement numerical schemes valid only for specific classes of processes.
        Then it suffices to implement the scheme and subclass this method to add the corresponding
        'if' statement, without rewriting the entire `trajectory` method.

        The implemented schemes are the following:

        - Euler-Maruyama [1]_ [2]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n`.

        It is the straightforward generalization to SDEs of the Euler method for ODEs.

        The Euler-Maruyama method has strong order 0.5 and weak order 1.
        """
        method = kwargs.get('method', 'euler')
        dt = kwargs.get('dt', self.default_dt)
        if method in ('euler', 'euler-maruyama', 'em'):
            x = self._euler_maruyama(x, t, w, dt)
        elif method == 'milstein':
            x = self._milstein(x, t, w, dt)
        else:
            raise NotImplementedError('SDE integration error: Numerical scheme not implemented')
        return x


    @pseudorand
    def trajectory(self, x0, t0, **kwargs):
        r"""
        Integrate the SDE with given initial condition.

        Parameters
        ----------
        x0: ndarray
            The initial position (in :math:`\mathbb{R}^n`).
        t0: float
            The initial time.

        Keyword Arguments
        -----------------
        dt: float
            The time step
            (default 0.1, unless overridden by a subclass).
        T: float
            The time duration of the trajectory (default 10).
        finite: bool
            Filter finite values before returning trajectory (default False).

        Returns
        -------
        t, x: ndarray, ndarray
            Time-discrete sample path for the stochastic process with initial conditions (t0, x0).
            The array t contains the time discretization and x the value of the sample path
            at these instants.
        """
        x = [x0]
        dt = kwargs.pop('dt', self.default_dt) # Time step
        time = kwargs.get('T', 10.0)   # Total integration time
        if dt < 0:
            raise ValueError("Timestep dt cannot be negative")
        precision = kwargs.pop('precision', np.float32)
        num = int(time/dt)+1
        tarray = np.linspace(t0, t0+time, num=num, dtype=precision)
        trajectory_shape = (num,len(x0)) if self.dimension > 1 else (num,)
        x = np.full(trajectory_shape, x0, dtype=precision)
        if 'brownian_path' in kwargs:
            tw, w = kwargs.pop('brownian_path')
            dw = np.diff(w, axis=0)
            deltat = tw[1]-tw[0]
            ratio = int(np.rint(dt/deltat)) # Both int and rint needed here ?
            dw = dw[:((num-1)*ratio)] # Trim noise vector if sequence w too long
        else:
            deltat = kwargs.pop('deltat', dt)
            ratio = int(np.rint(dt/deltat))
            brownian_path_shape = ((num-1)*ratio,len(x0)) if self.dimension > 1 else ((num-1)*ratio,)
            dw = np.random.normal(0, np.sqrt(deltat), size=brownian_path_shape)

            # As of numpy 1.18, random.normal does not support setting the dtype of
            # the returned array (https://github.com/numpy/numpy/issues/10892).
            # We cast dw to the same type returned by the diffusion function to prevent a numba
            # TypingError in self._euler_maruyama.
            # See issue https://github.com/cbherbert/stochrare/issues/14
            returned_array = self.diffusion(x[0], tarray[0])
            dw = dw.astype(returned_array.dtype)

        dw = self._integrate_brownian_path(dw, num, ratio)
        x = self.integrate_sde(x, tarray, dw, dt=dt, **kwargs)
        if kwargs.get('finite', False):
            tarray = tarray[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return tarray, x


    def _euler_maruyama(self, x, t, w, dt):
        if self.dimension > 1:
            return self._euler_maruyama_multidim(x, t, w, dt, self.drift, self.diffusion)
        else:
            return self._euler_maruyama_1d(x, t, w, dt, self.drift, self.diffusion)


    @staticmethod
    @jit(nopython=True)
    def _euler_maruyama_multidim(x, t, w, dt, drift, diffusion):
        for index in range(len(w)):
            wn = w[index]
            xn = x[index]
            tn = t[index]
            x[index+1] = xn + drift(xn, tn)*dt + np.dot(diffusion(xn, tn), wn)
        return x


    @staticmethod
    @jit(nopython=True)
    def _euler_maruyama_1d(x, t, w, dt, drift, diffusion):
        for index in range(len(w)):
            wn = w[index]
            xn = x[index]
            tn = t[index]
            x[index+1] = xn + drift(xn, tn)*dt + diffusion(xn, tn)*wn
        return x


    @method1d
    def _milstein(self, x, t, w, dt):
        for index, wn in enumerate(w):
            xn = x[index]
            tn = t[index]
            a = self.drift(xn, tn)
            b = self.diffusion(xn, tn)
            db = derivative(self.diffusion, xn, dx=1e-6, args=(tn,))
            x[index+1] = xn + (a-0.5*b*db)*dt + b*wn + 0.5*b*db*wn**2
        return x


    @pseudorand
    def trajectory_generator(self, x0, t0, nsteps, **kwargs):
        r"""
        Integrate the SDE with given initial condition, generator version.

        Parameters
        ----------
        x0: ndarray
            The initial position (in :math:`\mathbb{R}^n`).
        t0: float
            The initial time.
        nsteps: int
            The number of samples to generate.

        Keyword Arguments
        -----------------
        dt: float
            The time step, forwarded to the :meth:`update` routine
            (default 0.1, unless overridden by a subclass).
        observable: function with two arguments
            Time-dependent observable :math:`O(x, t)` to compute (default :math:`O(x, t)=x`)

        Yields
        -------
        t, y: ndarray, ndarray
            Time-discrete sample path (or observable) for the stochastic process with initial
            conditions (t0, x0).
            The array t contains the time discretization and y=O(x, t) the value of the observable
            (it may be the stochastic process itself) at these instants.
        """
        x = x0
        t = t0
        dt = kwargs.get('dt', self.default_dt) # Time step
        obs = kwargs.get('observable', lambda x, t: x)
        yield t0, obs(x0, t0)
        for _ in range(nsteps):
            t = t + dt
            x = self.update(x, t, dt=dt)
            yield t, obs(x, t)


    def trajectory_conditional(self, x0, t0, pred, **kwargs):
        r"""
        Compute sample path satisfying arbitrary condition.

        Parameters
        ----------
        x0: float
            The initial position.
        t0: float
            The initial time.
        pred: function with two arguments
            The predicate to select trajectories.

        Keyword Arguments
        -----------------
        dt: float
            The time step, forwarded to the :meth:`update` routine
            (default 0.1, unless overridden by a subclass).
        T: float
            The time duration of the trajectory (default 10).
        finite: bool
            Filter finite values before returning trajectory (default False).

        Returns
        -------
        t, x: ndarray, ndarray
            Time-discrete sample path for the stochastic process with initial conditions (t0, x0).
            The array t contains the time discretization and x the value of the sample path
            at these instants.
        """
        while True:
            t, x = self.trajectory(x0, t0, **kwargs)
            if pred(t, x):
                break
        return t, x


    def sample_mean(self, x0, t0, nsteps, nsamples, **kwargs):
        r"""
        Compute the sample mean of a time dependent observable, conditioned on initial conditions.

        Parameters
        ----------
        x0: ndarray
            The initial position (in :math:`\mathbb{R}^n`).
        t0: float
            The initial time.
        nsteps: int
            The number of samples in each sample path.
        nsamples: int
            The number of sample paths in the ensemble.

        Keyword Arguments
        -----------------
        dt: float
            The time step, forwarded to the :meth:`update` routine
            (default 0.1, unless overridden by a subclass).
        observable: function with two arguments
            Time-dependent observable :math:`O(x, t)` to compute (default :math:`O(x, t)=x`)

        Yields
        -------
        t, y: ndarray, ndarray
            Time-discrete ensemble mean for the observable, conditioned on the initial
            conditions (t0, x0).
            The array t contains the time discretization and :math:`y=\mathbb{E}[O(x, t)]`
            the value of the sample mean of the observable (it may be the stochastic process itself)
            at these instants.
        """
        for ensemble in zip(*[self.trajectory_generator(x0, t0, nsteps, **kwargs)
                              for _ in range(nsamples)]):
            time, obs = zip(*ensemble)
            yield np.average(time, axis=0), np.average(obs, axis=0)


    def empirical_vector(self, x0, t0, nsamples, *args, **kwargs):
        r"""
        Empirical vector at given times.

        Parameters
        ----------
        x0 : float
            Initial position.
        t0 : float
            Initial time.
        nsamples : int
            The size of the ensemble.
        *args : variable length argument list
            The times at which we want to estimate the empirical vector.

        Keyword Arguments
        -----------------
        **kwargs :
            Keyword arguments forwarded to :meth:`trajectory` and to :meth:`numpy.histogram`.

        Yields
        ------
        t, pdf, bins : float, ndarray, ndarray
            The time and histogram of the stochastic process at that time.

        Notes
        -----
        This method computes the empirical vector, or in other words, the relative frequency of the
        stochastic process at different times, conditioned on the initial condition.
        At each time, the empirical vector is a random vector.
        It is an estimator of the transition probability :math:`p(x, t | x_0, t_0)`.
        """
        hist_kwargs_keys = ('bins', 'range', 'weights') # hard-coded for now, we might use inspect
        hist_kwargs = {key: kwargs[key] for key in kwargs if key in hist_kwargs_keys}
        def traj_sample(x0, t0, *args, **kwargs):
            if 'brownian_path' in kwargs:
                tw, w = kwargs.get('brownian_path')
                dt = kwargs.get('dt', self.default_dt)
                offset=0
            for i, tsample in enumerate(args):
                if 'brownian_path' in kwargs:
                    deltat = tw[1]-tw[0]
                    num = int((tsample-t0)/deltat)+1
                    brownian_path_chunk = (tw[offset:num+offset], w[offset:num+offset])
                    offset = num + offset - 1
                    kwargs.update({'brownian_path': brownian_path_chunk})
                t, x = self.trajectory(x0, t0, T=tsample-t0, **kwargs)
                t0 = t[-1]
                x0 = x[-1]
                yield tsample, x0


        brownian_paths = kwargs.pop('brownian_paths', None)
        traj_ensemble = []
        for sample in range(nsamples):
            if brownian_paths:
                kwargs.update({'brownian_path': brownian_paths[sample]})
            traj_ensemble.append(traj_sample(x0, t0, *args, **kwargs))

        for ensemble in zip(*traj_ensemble):
            time, obs = zip(*ensemble)
            yield (time[0], ) + np.histogram(obs, density=True, **hist_kwargs)


    @method1d
    def instantoneq(self, t, Y):
        r"""
        Equations of motion for instanton dynamics.

        Parameters
        ----------
        t: float
            The time.
        Y: ndarray or list
            Vector with two elements: x=Y[0] the position and p=Y[1] the impulsion.

        Returns
        -------
        xdot, pdot: ndarray (size 2)
            The right hand side of the Hamilton equations.

        Notes
        -----
        These are the Hamilton equations corresponding to the following action:
        :math:`A=1/2 \int ((\dot{x}-b(x, t))/sigma(x, t))^2 dt`, i.e.
        :math:`\dot{x}=\sigma(x,t)^2*p+b(x, t)` and
        :math:`\dot{p}=-\sigma(x, t)*\sigma'(x, t)*p^2-b'(x, t)*p`.

        The Hamiltonian is :math:`H=\sigma^2(x, t)*p^2/2+b(x, t)*p`.

        Note that these equations include the diffusion coefficient, unlike those we use in the case
        of a constant diffusion process `ConstantDiffusionProcess1D`.
        Hence, for constant diffusion coefficients, the two only coincide when D=1.
        Otherwise, it amounts at a rescaling of the impulsion.
        """
        x = Y[0]
        p = Y[1]
        dbdx = derivative(self.drift, x, dx=1e-6, args=(t, ))
        dsigmadx = derivative(self.diffusion, x, dx=1e-6, args=(t, ))
        return np.array([p*self.diffusion(x, t)**2+self.drift(x, t),
                         -p**2*self.diffusion(x, t)*dsigmadx-p*dbdx])


    @method1d
    def instantoneq_jac(self, t, Y):
        r"""
        Jacobian of the equations of motion for instanton dynamics.

        Parameters
        ----------
        t: float
            The time.
        Y: ndarray or list
            Vector with two elements: x=Y[0] the position and p=Y[1] the impulsion.

        Returns
        -------
        xdot, pdot: ndarray (shape (2, 2))
            The Jacobian of the right hand side of the Hamilton equations, i.e.
            :math:`[[d\dot{x}/dx, d\dot{x}/dp], [d\dot{p}/dx, d\dot{p}/dp]]`.

        Notes
        -----
        These are the Hamilton equations corresponding to the following action:
        :math:`A=1/2 \int ((\dot{x}-b(x, t))/sigma(x, t))^2 dt`, i.e.
        :math:`\dot{x}=\sigma(x,t)^2*p+b(x, t)` and
        :math:`\dot{p}=-\sigma(x, t)*\sigma'(x, t)*p^2-b'(x, t)*p`.

        The Hamiltonian is :math:`H=\sigma^2(x, t)*p^2/2+b(x, t)*p`.

        Note that these equations include the diffusion coefficient, unlike those we use in the case
        of a constant diffusion process `ConstantDiffusionProcess1D`.
        Hence, for constant diffusion coefficients, the two only coincide when D=1.
        Otherwise, it amounts at a rescaling of the impulsion.
        """
        x = Y[0]
        p = Y[1]
        dbdx = derivative(self.drift, x, dx=1e-6, args=(t, ))
        d2bdx2 = derivative(self.drift, x, n=2, dx=1e-5, args=(t, ))
        sigma = self.diffusion(x, t)
        dsigmadx = derivative(self.diffusion, x, dx=1e-6, args=(t, ))
        d2sigmadx2 = derivative(self.diffusion, x, n=2, dx=1e-5, args=(t, ))
        return np.array([[dbdx+2*p*sigma*dsigmadx, sigma**2],
                         [-p*d2bdx2-p**2*(dsigmadx**2+sigma*d2sigmadx2), -dbdx-2*p*sigma*dsigmadx]])


    @method1d
    def _fpthsol(self, X, t, **kwargs):
        """ Analytic solution of the Fokker-Planck equation, when it is known.
        In general this is an empty method but subclasses corresponding to stochastic processes
        for which theoretical results exists should override it."""
        return NotImplemented


    @method1d
    @classmethod
    def trajectoryplot(cls, *args, **kwargs):
        """
        Plot 1D  trajectories.

        Parameters
        ----------
        *args : variable length argument list
        trajs: tuple (t, x)

        Keyword Arguments
        -----------------
        fig : matplotlig.figure.Figure
            Figure object to use for the plot. Create one if not provided.
        ax : matplotlig.axes.Axes
            Axes object to use for the plot. Create one if not provided.
        **kwargs :
            Other keyword arguments forwarded to matplotlib.pyplot.axes.

        Returns
        -------
        fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
            The figure.

        Notes
        -----
        This is just an interface to the function :meth:`stochrare.io.plot.trajectory_plot1d`.
        However, it may be overwritten in subclasses to systematically include elements to
        the plot which are specific to the stochastic process.
        """
        return plot.trajectory_plot1d(*args, **kwargs)


class ConstantDiffusionProcess(DiffusionProcess):
    r"""
    Diffusion processes, in arbitrary dimensions, with constant diffusion coefficient.

    It corresponds to the family of SDEs :math:`dx_t = F(x_t, t)dt + \sigma dW_t`,
    where :math:`F` is a time-dependent :math:`N`-dimensional vector field
    and :math:`W` the :math:`N`-dimensional Wiener process.
    The diffusion coefficient :math:`\sigma` is independent of the stochastic process
    (additive noise) and time, and we further assume that it is proportional to the identity matrix:
    all the components of the noise are independent.

    Parameters
    ----------
    vecfield : function with two arguments
        The vector field :math:`F(x, t)`.
    Damp : float
        The amplitude of the noise.
    dim : int
        The dimension of the system.

    Notes
    -----
    The diffusion coefficient is given by :math:`\sigma=\sqrt{2\text{Damp}}`.
    This convention leads to simpler expressions, for instance for the Fokker-Planck equations.
    """

    default_dt = 0.1

    def __init__(self, vecfield, Damp, dim, **kwargs):
        """
        vecfield: vector field, function of two variables (x,t)
        Damp: amplitude of the diffusion term (noise), scalar
        dim: dimension of the system

        In this class of stochastic processes, the diffusion matrix is proportional to identity.
        """
        DiffusionProcess.__init__(self, vecfield, (lambda x, t: np.sqrt(2*Damp)*np.eye(dim)),
                                  dim, **kwargs)
        self._D0 = Damp

    @property
    def diffusion(self):
        return self._diffusion

    @diffusion.setter
    def diffusion(self, diffusionnew):
        raise TypeError("ConstantDiffusionProcess objects do not allow setting the diffusion attribute")

    @property
    def D0(self):
        return self._D0

    @D0.setter
    def D0(self, D0new):
        self._D0 = D0new
        dim = self.dimension
        self._diffusion = jit(lambda x, t: np.sqrt(2*D0new)*np.eye(dim), nopython=True)

    def update(self, xn, tn, **kwargs):
        r"""
        Return the next sample for the time-discretized process.

        Parameters
        ----------
        xn: ndarray
            A n-dimensional vector (in :math:`\mathbb{R}^n`).
        tn: float
            The current time.

        Keyword Arguments
        ------------------
        dt : float
            The time step.
        dw : ndarray
            The brownian increment if precomputed.
            By default, it is generated on the fly from a Gaussian
            distribution with variance :math:`dt`.

        Returns
        -------
        x : ndarray
            The position at time tn+dt.

        See Also
        --------
        :meth:`DiffusionProcess.update` : for details about the Euler-Maruyama method.

        Notes
        -----
        This is the same as the :meth:`DiffusionProcess.update` method from the parent class
        :class:`DiffusionProcess`, except that a matrix product is no longer necessary.
        """
        dt = kwargs.get('dt', self.default_dt)
        if len(xn) != self.dimension:
            raise ValueError('Input vector does not have the right dimension.')
        dw = kwargs.get('dw', np.random.normal(0.0, np.sqrt(dt), self.dimension))
        return xn + self.drift(xn, tn)*dt+np.sqrt(2*self.D0)*dw


class OrnsteinUhlenbeck(ConstantDiffusionProcess):
    r"""
    The Ornstein-Uhlenbeck process, in arbitrary dimensions.

    It corresponds to the SDE :math:`dx_t = \theta(\mu-x_t)dt + \sqrt{2D} dW_t`,
    where :math:`\theta>0` and :math:`\mu \in \mathbb{R}^n` are arbitrary coefficients
    and :math:`D>0` is the amplitude of the noise.

    Parameters
    ----------
    mu : ndarray
        The expectation value.
    theta : float
        The inverse of the relaxation time.
    D : float
        The amplitude of the noise.
    dim : int
        The dimension of the system.

    Notes
    -----
    The Ornstein-Uhlenbeck process has been used to model many systems.
    It was initially introduced to describe the motion of a massive
    Brownian particle with friction [3]_ .
    It may also be seen as a diffusion process in a harmonic potential.

    Because many of its properties can be computed analytically, it provides a useful
    toy model for developing new methods.

    References
    ----------
    .. [3] G. E. Uhlenbeck and L. S. Ornstein, "On the theory of Brownian Motion".
           Phys. Rev. 36, 823–841 (1930).
    """
    def __init__(self, mu, theta, D, dim, **kwargs):
        super(OrnsteinUhlenbeck, self).__init__(lambda x, t: theta*(mu-x), D, dim, **kwargs)
        self._theta = theta
        self._mu = mu

    @property
    def drift(self):
        return self._drift

    @drift.setter
    def drift(self, driftnew):
        raise TypeError("OrnsteinUhlenbeck objects do not allow setting the drift attribute")

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, munew):
        self._mu = munew
        theta = self.theta
        self._drift = jit(lambda x, t: theta*(munew-x), nopython=True)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, thetanew):
        self._theta = thetanew
        mu = self.mu
        self._drift = jit(lambda x, t: thetanew*(mu-x), nopython=True)

    def __str__(self):
        label = f"{self.dimension}D Ornstein-Uhlenbeck process"
        eq = "dx_t = theta(mu-x_t)dt + sqrt(2D) dW_t"
        return f"{label}: {eq}, with theta={self.theta}, mu={self.mu} and D={self.D0}."

    def potential(self, X):
        r"""
        Compute the potential from which the force derives.

        Parameters
        ----------
        X : ndarray, shape (npts, self.dimension)
            The points where we want to compute the potential

        Returns
        -------
        V : float, shape (npts, )
            The potential from which the force derives, at the given points.

        Notes
        -----
        Not all diffusion processes derive from a potential, but the Ornstein Uhlenbeck does.
        It is a gradient system, with a quadratic potential:
        :math:`dx_t = -\nabla V(x_t)dt + \sqrt{2D} dW_t`, with
        :math:`V(x) = \theta(\mu-x)^2/2`.
        """
        return np.array([self.theta*np.dot(self.mu-y, self.mu-y)/2 for y in X])

class Wiener(OrnsteinUhlenbeck):
    r"""
    The Wiener process, in arbitrary dimensions.

    Parameters
    ----------
    dim : int
        The dimension of the system.
    D : float, optional
        The amplitude of the noise (default is 1).

    Notes
    -----
    The Wiener process is a central object in the theory or stochastic processes,
    both from a mathematical point of view and for its applications in different scientific fields.
    We refer to classical textbooks for more information about the Wiener process
    and Brownian motion.
    """
    def __init__(self, dim, D=1, **kwargs):
        super(Wiener, self).__init__(0, 0, D, dim, **kwargs)

    @classmethod
    def potential(cls, X):
        r"""
        Compute the potential from which the force derives.

        Parameters
        ----------
        X : ndarray, shape (npts, self.dimension)
            The points where we want to compute the potential.

        Returns
        -------
        V : float, shape (npts, )
            The potential from which the force derives, at the given points.

        Notes
        -----
        The Wiener Process is a trivial gradient system, with vanishing potential.
        It is useless (and potentially source of errors) to call the general potential routine,
        so we just return zero directly.
        """
        return np.zeros(len(X))
