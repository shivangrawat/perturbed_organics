import torch
from .ORGaNICs3D import ORGaNICs3D
from gaussian_rect.utils.util_funs import dynm_fun
from gaussian_rect.spectrum_general.sim_spectrum import sim_solution
from gaussian_rect.utils.util_funs import dynm_fun, make_spike_train
import os


class ORGaNICs3D0(ORGaNICs3D):
    def __init__(self, params):
        super().__init__(params)

        """Initialize the circuit"""
        self.dim = self.calculate_dim()
        self.initialize_circuit()
        self.make_noise_mats()


class ORGaNICs3D1(ORGaNICs3D):
    def __init__(self, params):
        super().__init__(params)
        """
        This model implements the 3D ORGaNICs model with noise in input gain and a 
        dynamical equation for the firing variables.
        """
        self._Nb = params["N_b"]
        self._tauB = params["tauB"]

        """Type of noise"""
        self._noise_type = "multiplicative"

        """Baseline firing rate (Hz)"""
        self.max_firing = params["max_firing"]

        """Initialize the circuit"""
        self.dim = self.calculate_dim()
        self.initialize_circuit()
        self.make_noise_mats()

    @property
    def Nb(self):
        return self._Nb

    @Nb.setter
    def Nb(self, Nb):
        if Nb > 0:
            self._Nb = Nb
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Number of neurons must be a positive integer")

    @property
    def tauB(self):
        return self._tauB

    @tauB.setter
    def tauB(self, tauB):
        if tauB > 0:
            self._tauB = tauB
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def noise_type(self):
        return self._noise_type

    @noise_type.setter
    def noise_type(self, noise_type):
        if not noise_type == "multiplicative":
            raise ValueError("Noise type for the model can only be multiplicative.")

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        if 0 <= c <= 1:
            self._c = c
            self.input = self.make_input()
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Input grating contrast must be between 0 and 1")

    def get_instance_variables(self):
        return (
            self._Ny,
            self._Na,
            self._Nu,
            self._Nb,
            self._eta,
            self._tauY,
            self._tauA,
            self._tauU,
            self._tauB,
            self._sigma,
            self._alpha,
            self._b0,
            self._norm_band,
            self._input_dim,
            self._c,
            self._grating_angle,
        )

    def calculate_dim(self):
        return self._Ny + self._Na + self._Nu + self._Nb + self._Ny

    def make_Ly(self, t, x):
        """
        This function defines the most general form of the noise matrix.
        :return: The N x N noise matrix.
        """
        y = x[0 : self.Ny]
        L = self.eta * torch.eye(self.dim)
        L[
            (self.Ny + self.Na + self.Nu + self.Nb) :,
            (self.Ny + self.Na + self.Nu + self.Nb) :,
        ] = torch.sqrt(self.max_firing * y**2) * torch.eye(self.Ny)
        return L

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.zeros(self.dim)
        D[0 : self.Ny] = 1 / self.tauY
        D[self.Ny : (self.Ny + self.Na)] = 1 / self.tauA
        D[(self.Ny + self.Na) : (self.Ny + self.Na + self.Nu)] = 1 / self.tauU
        D[(self.Ny + self.Na + self.Nu) : (self.Ny + self.Na + self.Nu + self.Nb)] = (
            1 / self.tauB
        )
        D[(self.Ny + self.Na + self.Nu + self.Nb) :] = 1 / self.tauY
        return torch.diag(D)

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the ring ORGaNICs model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension
        y = x[0 : self.Ny]
        a = x[self.Ny : (self.Ny + self.Na)]
        u = x[(self.Ny + self.Na) : (self.Ny + self.Na + self.Nu)]
        b = x[(self.Ny + self.Na + self.Nu) : (self.Ny + self.Na + self.Nu + self.Nb)]
        yf = x[(self.Ny + self.Na + self.Nu + self.Nb) :]
        cc = ((self.sigma * self.b0) / (1 + self.b0)) ** 2
        u_plus = self.f(u)
        dydt = (1 / self.tauY) * (
            -y
            + (b / (1 + b)) * (self.Wzx @ self.input)
            + (1 / (1 + a)) * (self.Wyy @ y)
        )
        dudt = (1 / self.tauU) * (
            -u + (u / u_plus**2) * (self.Wuy @ (u_plus**2 * y**2) + cc)
        )
        dadt = (1 / self.tauA) * (
            -a + a * u_plus + u_plus + self.alpha * self.tauU * dudt
        )
        dbdt = (1 / self.tauB) * (-b + self.b0)
        dyfdt = (1 / self.tauY) * (-yf + self.max_firing * y**2)
        return torch.cat((dydt, dadt, dudt, dbdt, dyfdt))


class ORGaNICs3D2(ORGaNICs3D):
    def __init__(self, params):
        super().__init__(params)
        """
        This model implements the 3D ORGaNICs model with noise in input gain and Poisson
        noise for the variable representing firing rate. 
        """
        self._Nb = params["N_b"]
        self._tauB = params["tauB"]

        """Type of noise"""
        self._noise_type = "additive"
        # The reason why it's additive is because we're adding Poisson noise after
        # which is the term that carries the multiplicative noise

        """Baseline firing rate (Hz)"""
        self.max_firing = params["max_firing"]

        """Initialize the circuit"""
        # Note in this model self.dim is the number of variables we have to simulate
        # before adding the poisson spike trains
        self.dim = self.calculate_dim()
        self.initialize_circuit()
        self.make_noise_mats()

    @property
    def Nb(self):
        return self._Nb

    @Nb.setter
    def Nb(self, Nb):
        if Nb > 0:
            self._Nb = Nb
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Number of neurons must be a positive integer")

    @property
    def tauB(self):
        return self._tauB

    @tauB.setter
    def tauB(self, tauB):
        if tauB > 0:
            self._tauB = tauB
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def noise_type(self):
        return self._noise_type

    @noise_type.setter
    def noise_type(self, noise_type):
        if not noise_type == "multiplicative":
            raise ValueError("Noise type for the model can only be multiplicative.")

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        if 0 <= c <= 1:
            self._c = c
            self.input = self.make_input()
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Input grating contrast must be between 0 and 1")

    def get_instance_variables(self):
        return (
            self._Ny,
            self._Na,
            self._Nu,
            self._Nb,
            self._eta,
            self._tauY,
            self._tauA,
            self._tauU,
            self._tauB,
            self._sigma,
            self._alpha,
            self._b0,
            self._norm_band,
            self._input_dim,
            self._c,
            self._grating_angle,
        )

    def calculate_dim(self):
        return self._Ny + self._Na + self._Nu + self._Nb

    def make_Ly(self, t, x):
        """
        This function defines the most general form of the noise matrix.
        :return: The N x N noise matrix.
        """
        L = self.eta * torch.eye(self.dim)
        return L

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.zeros(self.dim)
        D[0 : self.Ny] = 1 / self.tauY
        D[self.Ny : (self.Ny + self.Na)] = 1 / self.tauA
        D[(self.Ny + self.Na) : (self.Ny + self.Na + self.Nu)] = 1 / self.tauU
        D[(self.Ny + self.Na + self.Nu) :] = 1 / self.tauB
        return torch.diag(D)

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the ring ORGaNICs model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension
        y = x[0 : self.Ny]
        a = x[self.Ny : (self.Ny + self.Na)]
        u = x[(self.Ny + self.Na) : (self.Ny + self.Na + self.Nu)]
        b = x[(self.Ny + self.Na + self.Nu) :]
        cc = ((self.sigma * self.b0) / (1 + self.b0)) ** 2
        u_plus = self.f(u)
        dydt = (1 / self.tauY) * (
            -y
            + (b / (1 + b)) * (self.Wzx @ self.input)
            + (1 / (1 + a)) * (self.Wyy @ y)
        )
        dudt = (1 / self.tauU) * (
            -u + (u / u_plus**2) * (self.Wuy @ (u_plus**2 * y**2) + cc)
        )
        dadt = (1 / self.tauA) * (
            -a + a * u_plus + u_plus + self.alpha * self.tauU * dudt
        )
        dbdt = (1 / self.tauB) * (-b + self.b0)
        return torch.cat((dydt, dadt, dudt, dbdt))

    def sde_simulation(self, n_points=int(1e5), time=10, dt=1e-4):
        """
        This function simulates the SDE model and adds Poisson noise on top of it.
        Note that the dt is for the SDE simulation.
        :return: The simulation of the SDE model.
        """
        sim_obj = sim_solution(self)
        t = torch.linspace(0, time, n_points)
        sol_sde = sim_obj.simulate_sde(t, n_points, time, dt)

        fr = self.max_firing * sol_sde[:, 0 : self.Ny] ** 2

        spike_train = make_spike_train(fr, dt=time / n_points)

        filtered_fr = torch.zeros_like(fr)

        for i in range(1, n_points):
            filtered_fr[i, :] = filtered_fr[i - 1, :] + (dt / self.tauY) * (
                -filtered_fr[i - 1, :] + spike_train[i - 1, :]
            )

        sol_sde = torch.cat((sol_sde, filtered_fr), dim=1)
        return spike_train, sol_sde, t


class ORGaNICs3Doldf(ORGaNICs3D):
    def __init__(self, params):
        super().__init__(params)

        """Initialize the circuit"""
        self.dim = self.calculate_dim()
        self.initialize_circuit()
        self.make_noise_mats()

    @staticmethod
    def f(u):
        return torch.relu(u)

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the ring ORGaNICs model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension
        y = x[0 : self.Ny]
        a = x[self.Ny : (self.Ny + self.Na)]
        u = x[(self.Ny + self.Na) :]
        cc = ((self.sigma * self.b0) / (1 + self.b0)) ** 2
        u_plus = self.f(u)
        dydt = (1 / self.tauY) * (
            -y
            + (self.b0 / (1 + self.b0)) * (self.Wzx @ self.input)
            + (1 / (1 + a)) * (self.Wyy @ y)
        )
        dudt = (1 / self.tauU) * (-u + (self.Wuy @ (u * y**2) + cc * u / u_plus**2))
        dadt = (1 / self.tauA) * (
            -a + a * u_plus + u_plus + self.alpha * self.tauU * dudt
        )
        return torch.cat((dydt, dadt, dudt))
