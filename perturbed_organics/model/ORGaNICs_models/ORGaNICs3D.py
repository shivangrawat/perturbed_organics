"""We define the class for simulating the ORGaNICs ring model using 3D ORGaNICs"""
import torch
from torch.func import grad
from .ORGaNICs import ORGaNICs
from gaussian_rect.utils.util_funs import dynm_fun
import os


class ORGaNICs3D(ORGaNICs):
    def __init__(self, params):
        super().__init__(params)
        """
        This is a prototypical 3D ORGaNICs model.
        """
        self._Ny = params['N_y']
        self._Na = params['N_a']
        self._Nu = params['N_u']

        """Noise strength"""
        self._eta = params['eta']

        """Time constants"""
        self._tauY = params['tauY']
        self._tauA = params['tauA']
        self._tauU = params['tauU']

        """More model parameters"""
        self._alpha = params['alpha']

        """Make the weight matrices"""
        self.Wuy = None

    @property
    def Ny(self):
        return self._Ny

    @Ny.setter
    def Ny(self, Ny):
        if Ny > 0:
            self._Ny = Ny
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Number of neurons must be a positive integer")

    @property
    def Na(self):
        return self._Na

    @Na.setter
    def Na(self, Na):
        if Na > 0:
            self._Na = Na
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Number of neurons must be a positive integer")

    @property
    def Nu(self):
        return self._Nu

    @Nu.setter
    def Nu(self, Nu):
        if Nu > 0:
            self._Nu = Nu
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Number of neurons must be a positive integer")

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        if eta >= 0:
            self._eta = eta
            self.make_noise_mats()
        else:
            raise ValueError("Noise S.D. must be a positive float")

    @property
    def tauY(self):
        return self._tauY

    @tauY.setter
    def tauY(self, tauY):
        if tauY > 0:
            self._tauY = tauY
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def tauA(self):
        return self._tauA

    @tauA.setter
    def tauA(self, tauA):
        if tauA > 0:
            self._tauA = tauA
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def tauU(self):
        return self._tauU

    @tauU.setter
    def tauU(self, tauU):
        if tauU > 0:
            self._tauU = tauU
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha >= 0:
            self._alpha = alpha
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("High pass filter gain parameter must be > 0")

    @property
    def norm_band(self):
        return self._norm_band

    @norm_band.setter
    def norm_band(self, norm_band):
        if norm_band >= 0:
            self._norm_band = norm_band
            self.Wuy = ORGaNICs.make_Wuy(self.Ny, self.Nu, self.norm_band)
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Normalization bandwidth must be larger than 0")

    def get_instance_variables(self):
        return (self._Ny, self._Na, self._Nu, self._eta, self._tauY, self._tauA,
                self._tauU, self._sigma, self._alpha, self._b0, self._norm_band,
                self._input_dim, self._c, self._grating_angle)

    @staticmethod
    def f(u):
        return torch.sqrt(torch.relu(u))

    def f_grad(self, u):
        """Return the derivative of f(u)"""
        def sum_func(u):
            return torch.sum(self.f(u))
        return grad(sum_func)(u)

    def calculate_dim(self):
        return self._Ny + self._Na + self._Nu

    def initialize_circuit(self):
        """
        This function makes the input stimulus, the weight matrices and the jacobian
        corresponding to the system.
        :return: None
        """

        """Make the input stimulus"""
        self.input = self.make_input()

        """Make the feedforward and recurrent weight matrices"""
        self.Wzx, self.Wyy = ORGaNICs.make_weight_matrices(self.input_dim, self.Ny)

        """Make the normalization matrix"""
        self.Wuy = ORGaNICs.make_Wuy(self.Ny, self.Nu, self.norm_band)

        """Make the jacobian"""
        _ = self.jacobian_autograd()
        return None

    def make_Ly(self, t, x):
        """
        This function defines the most general form of the noise matrix.
        :param N: The number of units in the network.
        :param eta: The strength (coefficient) of the noise.
        :return: The N x N noise matrix.
        """
        return self.eta * torch.eye(self.dim)

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.zeros(self.dim)
        D[0:self.Ny] = 1 / self.tauY
        D[self.Ny:(self.Ny + self.Na)] = 1 / self.tauA
        D[(self.Ny + self.Na):] = 1 / self.tauU
        return torch.diag(D)

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the ring ORGaNICs model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension
        y = x[0:self.Ny]
        a = x[self.Ny:(self.Ny + self.Na)]
        u = x[(self.Ny + self.Na):]
        cc = ((self.sigma * self.b0) / (1 + self.b0)) ** 2
        u_plus = self.f(u)
        dydt = (1 / self.tauY) * (
                    -y + (self.b0 / (1 + self.b0)) * (self.Wzx @ self.input) + (
                        1 / (1 + a)) * (self.Wyy @ y))
        dudt = (1 / self.tauU) * (-u + (u / u_plus ** 2) * (self.Wuy @ (u_plus ** 2 * y ** 2) + cc))
        dadt = (1 / self.tauA) * (
                    -a + a * u_plus + u_plus + self.alpha * self.f_grad(u) * self.tauU * dudt)
        return torch.cat((dydt, dadt, dudt))

    def ss_norm(self):
        """
        This function calculates the norm of the steady state of the firing rates of the
        principal neurons using the analytical solution.
        :return: The steady state of the network.
        """
        z = self.Wzx @ self.input
        ss = z**2 / (self.sigma**2 + self.Wuy @ z**2)
        return ss
