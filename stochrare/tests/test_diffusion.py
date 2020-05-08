"""
Unit tests for the diffusion module.
"""
import unittest
import numpy as np
import stochrare.dynamics.diffusion as diffusion
from unittest.mock import patch

class TestDynamics(unittest.TestCase):
    def setUp(self):
        self.oup = diffusion.OrnsteinUhlenbeck(0, 1, 1, 2, deterministic=True)
        self.wiener = diffusion.Wiener(2, D=0.5, deterministic=True)
        self.wiener1 = diffusion.Wiener(1, D=0.5, deterministic=True)

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

    def test_potential(self):
        x = np.linspace(-1, 1)
        np.testing.assert_array_equal(diffusion.Wiener.potential(x), np.zeros_like(x))
        oup = diffusion.OrnsteinUhlenbeck(0, 1, 1, 1)
        np.testing.assert_allclose(oup.potential(x), 0.5*x**2)
        np.testing.assert_allclose(diffusion.DiffusionProcess.potential(oup, x, 0), 0.5*x**2)
        x = np.ones((10, 10))
        np.testing.assert_array_equal(self.wiener.potential(x), np.zeros(len(x)))


    def test_update(self):
        # Check that _euler_maruyama is called correctly
        pass

    def test_update_ConstantDiffusionProcess(self):
        for wienerD in (self.wiener, self.wiener1):
            dw = np.random.normal(size=wienerD.dimension)
            x = np.zeros(wienerD.dimension)
            np.testing.assert_array_equal(wienerD.update(x, 0, dw=dw), dw)
            np.testing.assert_array_equal(diffusion.DiffusionProcess.update(wienerD, x, 0, dw=dw),
                                          dw)

    @patch.object(diffusion.DiffusionProcess, "_euler_maruyama")
    def test_integrate_sde(self, mock_DiffusionProcess):
        x0=0;time=10;dt=0.1;
        num = int(time/dt)+1
        tarray = np.linspace(0, time, num=num)

        for dim in range(1,5):
            x = np.full((num, dim), x0)
            dw = np.random.normal(0, np.sqrt(dt), size=(num-1, dim))
            with self.subTest(dim=dim):
                model = diffusion.DiffusionProcess(
                    lambda x, t: 2*x,
                    lambda x, t: np.identity(dim),
                    dim,
                    deterministic=True,
                )
                model.integrate_sde(x, tarray, dw, dt=dt, method="euler")
                model._euler_maruyama.assert_called_with(x,
                                                         tarray,
                                                         dw,
                                                         dt,
                                                         model.drift,
                                                         model.diffusion
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


    def test_trajectory(self):
        dt_brownian = 1e-5
        diff = lambda x, t: np.array([[x[0], 0], [0, x[1]]], dtype=np.float32)
        model = diffusion.DiffusionProcess(lambda x, t: 2*x, diff, 2, deterministic=True)
        brownian_path = self.wiener.trajectory(np.array([0., 0.]), 0., T=0.1, dt=dt_brownian)
        traj_exact1 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 0])
        traj_exact2 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 1])
        traj = model.trajectory(np.array([1., 1.]), 0., T=0.1, dt=dt_brownian,
                                brownian_path=brownian_path, precision=np.float32)
        np.testing.assert_allclose(traj[1][:, 0], traj_exact1, rtol=1e-2)
        np.testing.assert_allclose(traj[1][:, 1], traj_exact2, rtol=1e-2)

    def test_trajectory_generator(self):
        traj = np.array([x for t, x in self.oup.trajectory_generator(np.array([0, 0]), 0,
                                                                     100, dt=0.01)])
        _, x = self.oup.trajectory(np.array([0, 0]), 0, dt=0.01, T=1)
        np.testing.assert_allclose(x, traj, rtol=1e-5)


    def test_euler_maruyama(self):
        x = diffusion.DiffusionProcess._euler_maruyama(
            np.array([1,2,3]),
            1.,
            0.7*np.ones(3),
            0.01,
            lambda x, t: 2*x,
            lambda x, t: np.diag(x) + t*np.eye(3),
        )
        np.testing.assert_almost_equal(x[0], 2.42)
        np.testing.assert_almost_equal(x[1], 4.14)
        np.testing.assert_almost_equal(x[2], 5.86)

        x = diffusion.DiffusionProcess._euler_maruyama(
            np.array([3,0,0]*4, dtype=np.float32).reshape(4,3),
            np.array([1,2,3]),
            1*np.ones((3,3)),
            1.,
            lambda x, t: 2*x,
            lambda x, t: np.diag(x) + t*np.eye(3),
        )
        np.testing.assert_allclose(x[1], np.array([13., 1., 1.]))
        np.testing.assert_allclose(x[2], np.array([54., 6., 6.]))
        np.testing.assert_allclose(x[3], np.array([219., 27., 27.]))


if __name__ == "__main__":
    unittest.main()
