"""
Unit tests for the diffusion module.
"""
import unittest
import inspect
import numpy as np
import stochrare.dynamics.diffusion as diffusion
import stochrare.dynamics.diffusion1d as diffusion1d
from unittest.mock import patch
from numba import jit

class TestDynamics(unittest.TestCase):
    def setUp(self):
        self.oup = diffusion.OrnsteinUhlenbeck(0, 1, 1, 2, deterministic=True)
        self.wiener = diffusion.Wiener(2, D=0.5, deterministic=True)
        self.wiener1 = diffusion1d.Wiener1D(D=0.5, deterministic=True)


        self.diffusions = []
        self.diffusions.append(diffusion.DiffusionProcess(lambda x, t: 2*x,
                                                          lambda x, t: 1,
                                                          1,
                                                          deterministic=True))
        self.diffusions.append(diffusion.DiffusionProcess(lambda x, t: 2*x,
                                                          lambda x, t: np.identity(2),
                                                          2,
                                                          deterministic=True))
        self.diffusions.append(diffusion.DiffusionProcess(lambda x, t: 2*x,
                                                          lambda x, t: np.identity(3),
                                                          3,
                                                          deterministic=True))


    def test_properties(self):
        self.assertEqual(self.oup.D0, 1)
        self.assertEqual(self.oup.mu, 0)
        self.assertEqual(self.oup.theta, 1)
        np.testing.assert_allclose(self.oup.diffusion(np.array([1, 1]), 0),
                                   np.array([[np.sqrt(2), 0], [0, np.sqrt(2)]]))
        self.oup.D0 = 0.5
        np.testing.assert_allclose(self.oup.diffusion(np.array([1, 1]), 0),
                                   np.array([[1, 0], [0, 1]]))
        np.testing.assert_allclose(self.oup.drift(np.array([1, 1]), 0), np.array([-1, -1]))
        self.oup.theta = 2
        np.testing.assert_allclose(self.oup.drift(np.array([1, 1]), 0), np.array([-2, -2]))
        self.oup.mu = np.array([1, 1])
        np.testing.assert_allclose(self.oup.drift(np.array([1, 1]), 0), np.array([0, 0]))
        self.oup.D0 = 1
        self.oup.theta = 1
        self.oup.mu = 0
        self.assertEqual(self.oup.dimension, 2)
        with self.assertRaises(ValueError):
            model = diffusion.DiffusionProcess(lambda x,t: x, lambda x,t: x, -1)


    def test_update(self):
        x = 0.0
        dw = np.random.normal(1)
        result = self.diffusions[0].update(x, 0, dw=dw)
        np.testing.assert_equal(result, dw)

        for model in self.diffusions[1:]:
            x = np.zeros(shape=model.dimension)
            dw = np.random.normal(size=model.dimension)
            result = model.update(x, 0, dw=dw)
            np.testing.assert_array_equal(result, dw)


    def test_potential(self):
        x = np.linspace(-1, 1)
        np.testing.assert_array_equal(diffusion.Wiener.potential(x), np.zeros_like(x))
        oup = diffusion.OrnsteinUhlenbeck(0, 1, 1, 1)
        np.testing.assert_allclose(oup.potential(x), 0.5*x**2)
        np.testing.assert_allclose(diffusion.DiffusionProcess.potential(oup, x, 0), 0.5*x**2)
        x = np.ones((10, 10))
        np.testing.assert_array_equal(self.wiener.potential(x), np.zeros(len(x)))


    @patch.object(diffusion.DiffusionProcess, "_euler_maruyama")
    def test_integrate_sde(self, mock_DiffusionProcess):
        x0=0;time=10;dt=0.1;
        num = int(time/dt)+1
        tarray = np.linspace(0, time, num=num)

        for dim in range(1,5):
            if dim==1:
                x = np.full((num,), x0)
                dw = np.random.normal(0, np.sqrt(dt), size=(num-1,))
                diff = lambda x, t: 1.
            else:
                x = np.full((num, dim), x0)
                dw = np.random.normal(0, np.sqrt(dt), size=(num-1,dim))
                diff = lambda x, t: np.identity(dim)
            model = diffusion.DiffusionProcess(
                lambda x, t: 2*x,
                diff,
                dim,
                deterministic=True,
            )
            with self.subTest(dim=dim):
                model.integrate_sde(x, tarray, dw, dt=dt, method="euler")
                model._euler_maruyama.assert_called_with(x,
                                                         tarray,
                                                         dw,
                                                         dt
                )
        self.assertEqual(model._euler_maruyama.call_count, 4)


    def test_integrate_sde_wrong_method(self):
        model = diffusion.DiffusionProcess(
                    lambda x, t: 2*x,
                    lambda x, t: np.identity(2),
                    2,
                    deterministic=True,
                )
        with self.assertRaises(NotImplementedError):
            model.integrate_sde(0, 0, 0, method="fancy method")


    def test_integrate_brownian_path(self):
        func = lambda x,t: x
        model = diffusion.DiffusionProcess(func, func, 2)
        num = 4
        ratio = 3

        dw_wrong_shape = np.array([range(1,11), range(11,1,-1)]).transpose()
        with self.assertRaises(ValueError):
            model._integrate_brownian_path(dw_wrong_shape, num, ratio)

        # 2d
        dw_correct_shape = np.array([range(1,10), range(10,1,-1)]).transpose()
        integrated_dw = model._integrate_brownian_path(dw_correct_shape, num, ratio)
        solution_array = np.array([[6,27],[15,18], [24,9]])
        np.testing.assert_array_equal(integrated_dw, solution_array)

        # 1d
        model = diffusion.DiffusionProcess(func, func, 1)
        dw_correct_shape = np.array(range(1,10))
        integrated_dw = model._integrate_brownian_path(dw_correct_shape, num, ratio)
        solution_array = np.array([6, 15, 24])
        np.testing.assert_array_equal(integrated_dw, solution_array)


    def test_trajectory_same_timestep(self):
        dt_brownian = 1e-5
        diff = lambda x, t: np.array([[x[0], 0], [0, x[1]]], dtype=np.float32)
        model = diffusion.DiffusionProcess(lambda x, t: 2*x, diff, 2, deterministic=True)

        # Check that ValueError is raised if negative timestep dt
        with self.assertRaises(ValueError):
            traj = model.trajectory(np.array([1., 1.]), 0., T=0.1, dt=-1)

        brownian_path = self.wiener.trajectory(np.array([0., 0.]), 0., T=0.1, dt=dt_brownian)
        traj_exact1 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 0])
        traj_exact2 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 1])
        traj = model.trajectory(np.array([1., 1.]), 0., T=0.1, dt=dt_brownian,
                                brownian_path=brownian_path, precision=np.float32)
        np.testing.assert_allclose(traj[1][:, 0], traj_exact1, rtol=1e-2)
        np.testing.assert_allclose(traj[1][:, 1], traj_exact2, rtol=1e-2)

    def test_trajectory_lower_timestep(self):
        dt_brownian = 1e-5
        diff = lambda x, t: np.array([[x[0], 0], [0, x[1]]], dtype=np.float32)
        model = diffusion.DiffusionProcess(lambda x, t: 2*x, diff, 2, deterministic=True)
        brownian_path = self.wiener.trajectory(np.array([0., 0.]), 0., T=0.1, dt=dt_brownian)
        traj_exact1 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 0])
        traj_exact2 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 1])
        traj = model.trajectory(np.array([1., 1.]), 0., T=0.1, dt=2*dt_brownian,
                                brownian_path=brownian_path, precision=np.float32)
        np.testing.assert_allclose(traj[1][:, 0], traj_exact1[::2], rtol=1e-2)
        np.testing.assert_allclose(traj[1][:, 1], traj_exact2[::2], rtol=1e-2)

    def test_trajectory_compute_brownian_path(self):
        dt_brownian = 1e-5
        for dtype in [np.float32, np.float64]:
            diff = lambda x, t: np.array([[x[0], 0], [0, x[1]]], dtype=dtype)
            model = diffusion.DiffusionProcess(lambda x, t: 2*x, diff, 2, deterministic=True)
            traj = model.trajectory(np.array([1., 1.]), 0., T=0.1, dt=dt_brownian, precision=dtype)

            brownian_path = self.wiener.trajectory(np.array([0., 0.]), 0., T=0.1, dt=dt_brownian)
            traj_exact1 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 0])
            traj_exact2 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 1])

            np.testing.assert_allclose(traj[1][:, 0], traj_exact1, rtol=1e-2)
            np.testing.assert_allclose(traj[1][:, 1], traj_exact2, rtol=1e-2)


    def test_trajectory_compute_brownian_path_lower_timestep(self):
        dt_brownian = 1e-5
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                diff = lambda x, t: np.array([[x[0], 0], [0, x[1]]], dtype=dtype)
                model = diffusion.DiffusionProcess(lambda x, t: 2*x, diff, 2, deterministic=True)
                traj = model.trajectory(np.array([1., 1.]), 0.,
                                        T=0.1,
                                        dt=2*dt_brownian,
                                        deltat=dt_brownian,
                                        precision=dtype,
                )

                brownian_path = self.wiener.trajectory(np.array([0., 0.]), 0., T=0.1, dt=dt_brownian)
                traj_exact1 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 0])
                traj_exact2 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 1])

                np.testing.assert_allclose(traj[1][:, 0], traj_exact1[::2], rtol=1e-2)
                np.testing.assert_allclose(traj[1][:, 1], traj_exact2[::2], rtol=1e-2)

    def test_trajectory_generator(self):
        traj = np.array([x for t, x in self.oup.trajectory_generator(np.array([0, 0]), 0,
                                                                     100, dt=0.01)])
        _, x = self.oup.trajectory(np.array([0, 0]), 0, dt=0.01, T=1)
        np.testing.assert_allclose(x, traj, rtol=1e-5)


    def test_trajectory_conditional(self):
        def pred(t,x):
            Xp = [np.dot(xi,xi) for xi in x[t>0.5]]
            Xm = [np.dot(xi,xi) for xi in x[t<0.5]]
            return np.mean(Xp) > np.mean(Xm)

        x0 = np.array([0.,0.])
        t, x = self.oup.trajectory_conditional(x0, 0, pred, T=1)

        Xp = [np.dot(xi,xi) for xi in x[t>0.5]]
        Xm = [np.dot(xi,xi) for xi in x[t<0.5]]
        self.assertTrue(np.mean(Xp) > np.mean(Xm))


    def test_empirical_vector(self):
        model = diffusion.DiffusionProcess(lambda x, t: 2 * x, lambda x, t: x, 1)
        x0 = 1.0
        t0 = 0.0
        dt_brownian = 0.01
        dt = 0.1
        nsamples = 1
        brownian_paths = [
            self.wiener1.trajectory(x0, t0, T=1, dt=dt_brownian)
            for sample in range(nsamples)
        ]

        refererence_trajectories = [
            model.trajectory(x0, t0, T=1, brownian_path=brownian_path, dt=dt)[1]
            for brownian_path in brownian_paths
        ]
        args = [0.3, 0.5, 1.0]
        empirical_vector_gen = model.empirical_vector(
            x0, t0, nsamples, *args, brownian_paths=brownian_paths, dt=dt
        )
        for i, (t, pdf, bins) in enumerate(empirical_vector_gen):
            self.assertEqual(t, args[i])
            index = int(t / dt) + 1
            expected_obs = [
                trajectory[index - 1] for trajectory in refererence_trajectories
            ]
            expected_pdf, expected_bins = np.histogram(expected_obs, density=True)
            np.testing.assert_allclose(pdf, expected_pdf)
            np.testing.assert_allclose(bins, expected_bins)


    def testinstantoneq(self):
        model = diffusion.DiffusionProcess(lambda x, t: 2*x, lambda x, t: x, 1)

        xvec = np.linspace(-1,1,10)
        pvec = np.linspace(-1,1,10)

        instantoneq = np.array([model.instantoneq(0, [x,p]) for x,p in zip(xvec, pvec)])
        # instantoneq =  [[-3.          3.        ]
        #                 [-2.0260631   2.0260631 ]
        #                 [-1.28257888  1.28257888]
        #                 [-0.7037037   0.7037037 ]
        #                 [-0.22359396  0.22359396]
        #                 [ 0.22359396 -0.22359396]
        #                 [ 0.7037037  -0.7037037 ]
        #                 [ 1.28257888 -1.28257888]
        #                 [ 2.0260631  -2.0260631 ]
        #                 [ 3.         -3.        ]]

        expected_xdot = (lambda x,p: x*x*p + 2*x)(xvec,pvec)
        expected_pdot = (lambda x,p: -x*p*p - 2*p)(xvec,pvec)
        np.testing.assert_allclose(instantoneq[:,0], expected_xdot)
        np.testing.assert_allclose(instantoneq[:,1], expected_pdot)


    def testinstantoneq_jac(self):
        model = diffusion.DiffusionProcess(lambda x, t: 2 * x, lambda x, t: x, 1)

        xvec = np.linspace(-1, 1, 10)
        pvec = np.linspace(-1, 1, 10)

        instantoneq_jac = [model.instantoneq_jac(0, [x, p]) for x, p in zip(xvec, pvec)]

        expected_J11 = lambda x, p: 2 * x * p + 2
        expected_J12 = lambda x, p: x * x
        expected_J21 = lambda x, p: -p * p
        expected_J22 = lambda x, p: -2 * x * p - 2

        expected = [
            [
                [expected_J11(x, p), expected_J12(x, p)],
                [expected_J21(x, p), expected_J22(x, p)],
            ]
            for x, p in zip(xvec, pvec)
        ]
        np.testing.assert_allclose(instantoneq_jac, expected, atol=1e-5)


    @unittest.skip("Method not yet implemented")
    def test_fpthsol(self):
        raise NotImplementedError


    def test_euler_maruyama(self):
        # Test DiffusionProcess._euler_maruyama in dimension 3
        x = diffusion.DiffusionProcess._euler_maruyama_multidim(
            np.array([3.,0,0]*4).reshape(4,3),
            np.array([1.,2.,3.]),
            1.*np.ones((3,3)),
            1.,
            jit(lambda x, t: 2*x, nopython=True),
            jit(lambda x, t: np.diag(x) + t*np.eye(3), nopython=True),
        )
        np.testing.assert_allclose(x[1], np.array([13., 1., 1.]))
        np.testing.assert_allclose(x[2], np.array([54., 6., 6.]))
        np.testing.assert_allclose(x[3], np.array([219., 27., 27.]))

        # Test DiffusionProcess._euler_maruyama in dimension 1
        x0 = 1.
        x = diffusion.DiffusionProcess._euler_maruyama_1d(
            np.full((4,), x0),
            np.array(range(3)),
            np.array([1.,2.,1.]),
            0.1,
            jit(lambda x, t: 2*x, nopython=True),
            jit(lambda x, t: x + t, nopython=True),
        )
        np.testing.assert_allclose(x, np.array([1, 2.2, 9.04, 21.888]))

    def test_milstein(self):
        model = diffusion.DiffusionProcess(lambda x, t: 2*x, lambda x, t: x + t, 1)
        x0 = 1.
        x = model._milstein(
            np.full((4,), x0),
            np.array(range(3)),
            np.array([1.,2.,1.]),
            0.1,
            )
        np.testing.assert_allclose(x, [1., 2.65, 17.5975, 49.53337501])


class TestOrnsteinUlhenbeckProcess(unittest.TestCase):
    def setUp(self):
        self.oup = diffusion.OrnsteinUhlenbeck(0, 1, 1, 2, deterministic=True)
        self.wiener = diffusion.Wiener(2, D=0.5, deterministic=True)
        self.wiener1 = diffusion.Wiener(1, D=0.5, deterministic=True)


    def test_update(self):
        for wienerD in (self.wiener, self.wiener1):
            dw = np.random.normal(size=wienerD.dimension)
            x = np.zeros(wienerD.dimension)
            np.testing.assert_array_equal(wienerD.update(x, 0, dw=dw), dw)


if __name__ == "__main__":
    unittest.main()
