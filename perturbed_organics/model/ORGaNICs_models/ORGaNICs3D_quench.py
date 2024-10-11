import torch
from .ORGaNICs3D import ORGaNICs3D
from perturbed_organics.utils.util_funs import dynm_fun
from perturbed_organics.spectrum_general.sim_spectrum import sim_solution
from .ORGaNICs import ORGaNICs
from perturbed_organics.utils.util_funs import dynm_fun, make_spike_train
from perturbed_organics.model._dyn_models import _dyn_models
import os


class ORGaNICs3D_quench(_dyn_models):
    def __init__(self, params, input, device=None):
        super().__init__(device)
        """
        This model implements the 3D ORGaNICs model with noise in input gain and a 
        dynamical equation for the firing variables.
        """
        self.size = input.shape[0]

        """Parameters of the model"""
        self.Ny = params["Ny"]
        self.Na = self.Ny
        self.Nu = self.Ny
        self.Nb = self.Ny

        """Partition of the dynamical system"""
        self.var_partition = (
            self.Ny,
            self.Ny,
            self.Na,
            self.Na,
            self.Nu,
            self.Nu,
            self.Nb,
        )

        """More model parameters"""
        self.sigma0 = params["sigma"]
        self.sigma = self.sigma0
        self.b = params["b"]
        self.b0 = params["b0"]
        self.delta_tau = params["delta_tau"]

        """Make the input stimulus"""
        self.input = input
        self.z = None

        """The weight matrices"""
        self.Wzx = None
        self.Wyy = None
        self.Wuy = None

        """Noise strength"""
        self.eta = params["eta"]
        self.eta_const = params["eta_const"]

        """Time constants"""
        self.tauY = params["tauY"]
        self.tauA = params["tauA"]
        self.tauU = params["tauU"]
        self.tauB = params["tauB"]
        self.tauS = params["tauS"]

        """More model parameters"""
        self.alpha = params["alpha"]

        """Type of noise"""
        self.noise_type = "multiplicative"

        """Baseline firing rate (Hz)"""
        self.max_firing = params["max_firing"]

        """Initialize the circuit"""
        self.dim = self.calculate_dim()
        self.initialize_circuit()
        self.make_noise_mats()
    
    @staticmethod
    def f(u):
        return torch.sqrt(torch.relu(u))

    def get_instance_variables(self):
        return (
            self.Ny,
            self.Na,
            self.Nu,
            self.Nb,
            self.eta,
            self.tauY,
            self.tauA,
            self.tauU,
            self.tauB,
            self.sigma,
            self.alpha,
            self.b0
        )

    def calculate_dim(self):
        return self.Ny + self.Ny + self.Na + self.Na + self.Nu + self.Nu + self.Nb

    def initialize_circuit(self):
        """
        This function makes the input stimulus, the weight matrices and the jacobian
        corresponding to the system.
        :return: None
        """
        """Make the feedforward and recurrent weight matrices"""
        self.Wzx, self.Wyy = ORGaNICs.make_weight_matrices(self.size, self.Ny)

        """Make the normalization matrix"""
        self.Wuy = ORGaNICs.make_Wuy(self.Ny, self.Nu, 1)

        """Make the input drive"""
        self.z = self.make_z()

        """Assign the steady state"""
        # self.ss = torch.cat(self.analytical_ss())

        """Make the jacobian"""
        smallest_tau = min(self.tauY, self.tauA, self.tauU, self.tauB, self.tauS)
        time = 1
        points = int(time / (0.1 * smallest_tau))
        _ = self.jacobian_autograd(time=time, points=points)
        return None
    
    @staticmethod
    def y_plus(y):
        return torch.relu(y) ** 2 + 1e-3

    @staticmethod
    def a_plus(a):
        return torch.relu(a) + 1e-3

    @staticmethod
    def u_plus(u):
        return torch.sqrt(torch.relu(u)) + 1e-3

    def make_noise_mats(self):
        """
        This function creates the noise vector for this case.
        :return: None
        """
        ss = torch.cat(self.analytical_ss())
        self.L = self.make_Ly(0, ss)
        self.D = self.make_D()
        self.S = torch.sqrt(self.D)
        return
    
    def make_z(self):
        z = self.Wzx @ self.input
        return z

    def make_Ly(self, t, x):
        """
        This function defines the most general form of the noise vector.
        Note we don't care about noisy simulation in this case.
        To get the steady-state noise, input the steady-state solution, 'x'.
        :return: This is a vector instead of a matrix because noise matrix is diagonal.
        """
        y = x[0 : self.Ny]
        a = x[self.Ny + self.Ny : (self.Ny + self.Ny + self.Na)]
        u = x[
            (self.Ny + self.Ny + self.Na + self.Na) : (
                self.Ny + self.Ny + self.Na + self.Na + self.Nu
            )
        ]

        varY = self.y_plus(y) * (
            1 + (self.tauY / self.delta_tau) / (self.a_plus(a) / (1 + self.a_plus(a)))
        )
        varA = self.a_plus(a) * (
            1 + (self.tauA / self.delta_tau) / (1 - self.u_plus(u))
        )
        varU = self.u_plus(u) * (
            1 + (self.tauU / self.delta_tau) / ((self.sigma * self.b0) ** 2 / u)
        )
        varB = self.b * torch.ones(self.Nb)

        tensor_list = (
            self.eta * torch.ones(self.Ny, device=self.device),
            self.eta_const * torch.sqrt(varY),
            self.eta * torch.ones(self.Na, device=self.device),
            self.eta_const * torch.sqrt(varA),
            self.eta * torch.ones(self.Nu, device=self.device),
            self.eta_const * torch.sqrt(varU),
            self.eta_const * torch.sqrt(varB),
        )
        return torch.diag(torch.cat(tensor_list))

    def make_D(self):
        """
        This function creates the D diagonal vector for the noise.
        Note that this is for a high-dimensional system, therefore,
        we only store the diagonal vector.
        :return: The D vector.
        """
        s = (
            1 / self.tauY,
            1 / self.tauS,
            1 / self.tauA,
            1 / self.tauS,
            1 / self.tauU,
            1 / self.tauS,
            1 / self.tauS,
        )
        tensor_list = []
        for length, scalar in zip(self.var_partition, s):
            tensor_list.append(torch.ones(length, device=self.device) * scalar)
        return torch.diag(torch.cat(tensor_list))

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the grid ORGaNICs model with
        modified gaussian rectification model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension

        # Define the different  based on self.var_partition
        y, y_plus, a, a_plus, u, u_plus, b_plus = torch.split(x, self.var_partition)
        dydt = (1 / self.tauY) * (
            -y + b_plus * self.z + (1 / (1 + a_plus)) * (self.Wyy @ (torch.sqrt(y_plus)))
        )
        dypdt = (1 / self.tauS) * (-y_plus + self.y_plus(y))
        dudt = (1 / self.tauU) * (
            -u + (self.sigma * self.b0) ** 2 + self.Wuy @ (u_plus**2 * y_plus)
        )
        dupdt = (1 / self.tauS) * (-u_plus + self.u_plus(u))
        dadt = (1 / self.tauA) * (
            -a + a * u_plus + u_plus + self.alpha * self.tauU * dudt
        )
        dapdt = (1 / self.tauS) * (-a_plus + self.a_plus(a))
        dbpdt = (1 / self.tauS) * (-b_plus + self.b)
        return torch.cat((dydt, dypdt, dadt, dapdt, dudt, dupdt, dbpdt))

    def analytical_ss(self):
        """
        This function returns the analytical steady-state solution of the system.
        This is done for all the variables in the system.
        """
        u = (self.sigma * self.b0) ** 2 + self.Wuy @ (torch.relu(self.b * self.z) ** 2)
        u_plus = self.u_plus(u)
        a = u_plus / (1 - u_plus)
        a_plus = self.a_plus(a)
        y = self.b * self.z / u_plus
        y_plus = self.y_plus(y)
        b_plus = torch.ones(self.Nb) * self.b
        return (y, y_plus, a, a_plus, u, u_plus, b_plus)
