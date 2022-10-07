#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube
# @Date:   05-03-2020
# @Email:  charles@goldspot.ca
# @Last modified by:   charles
# @Last modified time: 2020-03-19T11:47:29-04:00


import emcee
import numpy as np

from .cython_funcs import Decomp_cyth
from .cython_funcs import ColeCole_cyth
from .cython_funcs import Dias2000_cyth
from .cython_funcs import Shin2015_cyth

from . import utils
from . import plotlib
from pybert import FDIP
# the whole complex forward calculation is in pygimli.physics ert
from pygimli.physics import ert
from pygimli.physics.SIP.models import modelColeColeRho


class Inversion(plotlib.plotlib, utils.utils):
    """An abstract class to perform inversion of SIP data.

    This class is the base constructor for the PolynomialDecomposition and
    ColeCole classes.

    Args:
        filepath (:obj:`str`): The path to the file to perform inversion on.
        nwalkers (:obj:`int`): Number of walkers to use to explore the
            parameter space. Defaults to 32.
        nsteps (:obj:`int`): Number of steps to perform in the MCMC
            simulation. Defaults to 5000.
        headers (:obj:`int`): The number of header lines in the file.
            Defaults to 1.
        ph_units (:obj:`str`): The units of the phase shift measurements.
            Choices: 'mrad', 'rad', 'deg'. Defaults to 'mrad'.

    """

    def __init__(self, filepath, nwalkers=32, nsteps=5000, headers=1,
                 ph_units='mrad'):

        # Get arguments
        self.filepath = filepath
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.headers = headers
        self.ph_units = ph_units

        # Set default attributes
        self._p0 = None
        self._params = {}
        self.__fitted = False

        # Load data
        self._data = self.load_data(self.filepath, self.headers, self.ph_units)
    def _log_likelihood(self, theta, f, x, y, yerr):
        """Returns the conditional log-likelihood of the observations. """
        sigma2 = yerr**2
        fwd = f(theta, x)
        print(fwd.shape, y.shape)
        return -0.5*np.sum((y - fwd)**2 / sigma2 + 2*np.log(sigma2))

    def _log_prior(self, theta, bounds):
        """Returns the prior log-probability of the model parameters. """
        if not ((bounds[0] < theta).all() and (theta < bounds[1]).all()):
            return -np.inf
        else:
            return 0.0

    def _log_probability(self, theta, model, bounds, x, y, yerr):
        """Returns the Bayes numerator log-probability. """
        lp = self._log_prior(theta, bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta, model, x, y, yerr)

    def _check_if_fitted(self):
        """Checks if the model has been fitted. """
        if not self.fitted:
            raise AssertionError('Model is not fitted! Fit the model to a '
                                 'dataset before attempting to plot results.')

    def fit(self, p0=None, pool=None, moves=None):
        """Samples the posterior distribution to fit the model to the data.

        Args:
            p0 (:obj:`ndarray`): Starting parameter values. Should be a 2D
                array with shape (nwalkers, ndim). If None, random values will
                be uniformly drawn from the parameter bounds. Defaults to None.
            pool (:obj:`pool`, optional): A pool object from the
                Python multiprocessing library. See
                https://emcee.readthedocs.io/en/stable/tutorials/parallel/.
                Defaults to None.
            moves (:obj:`moves`, optional): A `emcee` Moves class (see
                https://emcee.readthedocs.io/en/stable/user/moves/). If None,
                the emcee algorithm `StretchMove` is used. Defaults to None.

        """
        self._p0 = p0
        # self._bounds = self.param_bounds
        self.ndim = self.param_bounds.shape[1]

        if self._p0 is None:
            self._p0 = np.random.uniform(*self.param_bounds,
                                         (self.nwalkers, self.ndim))

        model_args = (self.forward, self.param_bounds, self._data['w'],
                      self._data['zn'], self._data['zn_err'])

        self._sampler = emcee.EnsembleSampler(self.nwalkers,
                                              self.ndim,
                                              self._log_probability,
                                              args=model_args,
                                              pool=pool,
                                              moves=moves,
                                              )

        self._sampler.run_mcmc(self._p0, self.nsteps, progress=True)
        self.__fitted = True

    def get_chain(self, **kwargs):
        """Gets the MCMC chains from a fitted model.

        Keyword Args:
            discard (:obj:`int`): Number of steps to discard (burn-in period).
            thin (:obj:`int`): Thinning factor.
            flat (:obj:`bool`): Whether or not to flatten the walkers. If flat
                is False, the output chain will have shape (nsteps, nwalkers,
                ndim). If flat is True, the output chain will have shape
                (nsteps*nwalkers, ndim).

        Returns:
            :obj:`ndarray`: The MCMC chain(s).

        """
        self._check_if_fitted()
        return self._sampler.get_chain(**kwargs)

    @property
    def p0(self):
        """:obj:`ndarray`: Starting parameter values. Should be a 2D array with
            shape (nwalkers, ndim)."""
        return self._p0

    @property
    def params(self):
        """:obj:`dict`: Parameter names and their bounds."""
        return self._params

    @params.setter
    def params(self, var):
        self._params = var

    @property
    def sampler(self):
        """:obj:`EnsembleSampler`: A `emcee` sampler object (see
            https://emcee.readthedocs.io/en/stable/user/sampler/)."""
        self._check_if_fitted()
        return self._sampler

    @property
    def data(self):
        """:obj:`dict`: The input data dictionary."""
        return self._data

    @property
    def fitted(self):
        """:obj:`bool`: Whether the model has been fitted or not."""
        return self.__fitted

    @property
    def param_names(self):
        """:obj:`list` of :obj:`str`: Ordered names of the parameters."""
        return list(self.params.keys())

    @property
    def param_bounds(self):
        """:obj:`list` of :obj:`float`: Ordered bounds of the parameters."""
        return np.array(list(self.params.values())).T


class PolynomialDecomposition(Inversion):
    """A polynomial decomposition inversion scheme for SIP data.

    Args:
        *args: Arguments passed to the Inversion class.
        poly_deg (:obj:`int`): The polynomial degree to use for the
            decomposition. Defaults to 5.
        c_exp (:obj:`float`): The c-exponent to use for the decomposition
            scheme. 0.5 -> Warburg, 1.0 -> Debye. Defaults to 1.0.
        **kwargs: Additional keyword arguments passed to the Inversion class.

    """

    def __init__(self, *args, poly_deg=5, c_exp=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_exp = c_exp
        self.poly_deg = poly_deg

        # Define a range of relaxation time values for the RTD
        min_tau = np.floor(min(np.log10(1./self._data['w'])) - 1)
        max_tau = np.floor(max(np.log10(1./self._data['w'])) + 1)
        n_tau = 2*self._data['N']
        self.log_tau = np.linspace(min_tau, max_tau, n_tau)

        # Precompute the log_tau_i**i values for the polynomial approximation
        deg_range = list(range(self.poly_deg+1))
        self.log_taus = np.array([self.log_tau**i for i in deg_range])
        self.taus = 10**self.log_tau  # Accelerates sampling

        # Add polynomial decomposition parameters to dict
        self.params.update({'r0': [0.9, 1.1]})
        self.params.update({f'a{x}': [-1, 1] for x in deg_range})

        # self._bounds = np.array(self.param_bounds).T

    def forward(self, theta, w):
        """Returns a Polynomial Decomposition impedance.

        Args:
            theta (:obj:`ndarray`): Ordered array of R0, a_{poly_deg},
                a_{poly_deg-1}, ..., a_{0}. See
                https://doi.org/10.1016/j.cageo.2017.05.001.
            w (:obj:`ndarray`): Array of angular frequencies to compute the
                impedance for (w = 2*pi*f).

        """
        return Decomp_cyth(w, self.taus, self.log_taus, self.c_exp,
                           R0=theta[0], a=theta[1:])


class PeltonColeCole(Inversion):
    """A generalized ColeCole inversion scheme for SIP data.

    Args:
        *args: Arguments passed to the Inversion class.
        n_modes (:obj:`int`): The number of ColeCole modes to use for the
            inversion. Defaults to 1.
        **kwargs: Additional keyword arguments passed to the Inversion class.

    """

    def __init__(self, *args, n_modes=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_modes = n_modes

        # Add multi-mode ColeCole parameters to dict
        range_modes = list(range(self.n_modes))
        self.params.update({'r0': [0.9, 1.1]})
        self.params.update({f'm{i+1}': [0.0, 1.0] for i in range_modes})
        self.params.update({f'log_tau{i+1}': [-15, 5] for i in range_modes})
        self.params.update({f'c{i+1}': [0.0, 1.0] for i in range_modes})

        # self._bounds = np.array(self.param_bounds).T

    def forward(self, theta, w):
        """Returns a ColeCole impedance.

        Args:
            theta (:obj:`ndarray`): Ordered array of R0, m_{1}, ...,
                m_{n_modes}, log_tau_{1}, ..., log_tau_{n_modes}, c_{1}, ...,
                c_{n_modes}. See https://doi.org/10.1016/j.cageo.2017.05.001.
            w (:obj:`ndarray`): Array of angular frequencies to compute the
                impedance for (w = 2*pi*f).

        """
        return ColeCole_cyth(w,
                             R0=theta[0],
                             m=theta[1:1+self.n_modes],
                             lt=theta[1+self.n_modes:1+2*self.n_modes],
                             c=theta[1+2*self.n_modes:])


class Dias2000(Inversion):
    """A generalized ColeCole inversion scheme for SIP data.

    Args:
        *args: Arguments passed to the Inversion class.
        **kwargs: Additional keyword arguments passed to the Inversion class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add Dias parameters to dict
        self.params.update({'r0': [0.9, 1.1],
                            'm': [0, 1],
                            'log_tau': [-20, 0],
                            'eta': [0, 150],
                            'delta': [0, 1]})

        # self._bounds = np.array(self.param_bounds).T

    def forward(self, theta, w):
        """Returns a Dias (2000) impedance.

        Args:
            theta (:obj:`ndarray`): Ordered array of R0, m, log_tau, eta,
                delta. See https://doi.org/10.1016/j.cageo.2017.05.001.
            w (:obj:`ndarray`): Array of angular frequencies to compute the
                impedance for (w = 2*pi*f).

        """
        return Dias2000_cyth(w, *theta)


class Shin2015(Inversion):
    """A Shin (2015) inversion scheme for SIP data.

    Args:
        *args: Arguments passed to the Inversion class.
        **kwargs: Additional keyword arguments passed to the Inversion class.

    .. warning::
        The Shin model implementation is yielding unexpected results
        and needs to be reviewed.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add Dias parameters to dict
        self.params.update({'R1': [0.0, 1.0],
                            'R2': [0.0, 1.0],
                            'log_Q1': [-15, -13],
                            'log_Q2': [-7, -5],
                            'n1': [0, 1],
                            'n2': [0, 1],
                            })

        # self._bounds = np.array(self.param_bounds).T

    def forward(self, theta, w):
        """Returns a Shin (2015) impedance.

        Args:
            theta (:obj:`ndarray`): Ordered array of R1, R2, log_Q1, log_Q2,
                n1, n2.
            w (:obj:`ndarray`): Array of angular frequencies to compute the
                impedance for (w = 2*pi*f).

        """
        return Shin2015_cyth(w,
                             R=theta[:2],
                             log_Q=theta[2:4],
                             n=theta[4:]
                             )




    

class fdipPeltonN(Inversion):
    """A generalized ColeCole inversion scheme for SIP data.
    Args:
        *args: Arguments passed to the Inversion class.
        mesh: a two-dimensional mesh created with pygimli.meshtools
        frvec: (:obj:`list`) frequency vector
        data: data container created with pygimli.physics.ert.ertScheme
        n_cells: number of model cells =  ndim
        **kwargs: Additional keyword arguments passed to the Inversion class.
     """   
    def __init__(self, *args, mesh, frvec, data, n_cells=6, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_cells = n_cells
        # self.rhoaAllDatapoint = rhoaAllDatapoint
        # self.phiaAllDatapoint = phiaAllDatapoint
        # Add ColeCole parameters for each cell to dict
        range_n_cells = list(range(self.n_cells))
        self.params.update({'r0': [0.9, 1.1]})
        self.params.update({f'm{i}': [0.0, 1.0] for i in range_n_cells})
        self.params.update({f'log_tau{i}': [-15, 5] for i in range_n_cells})
        self.params.update({f'c{i}': [0.0, 1.0] for i in range_n_cells})

        # self._bounds = np.array(self.param_bounds).T
        self.mesh = mesh 
        self.fdip = FDIP(f=frvec, data=data) # we need to initialize the frequency vecctor and synthetic data vector (data contaider)


    def forward(self, model, f): # in forward calculation we need one argument which is the 'model vector'
        """Returns a ColeCole impedance.

        Args:
            mesh: 
                rhovec: ndarray of R0
            mvec: ndarray of m
            tauvec: ndarray of tau
            cvec: ndarray of c


        """
        # Unpack or subdivide the model vector (every model cell have 4 different 
        # parameters (rho, m, tau, c))
    
        rhovec, mvec, tauvec, cvec = np.reshape(model, [4, -1]) 
        # now it is ready to be used in simulate function
        return self.fdip.simulate(self.mesh, f, rhovec, mvec, tauvec, cvec)


class fdipPelton(Inversion):
    """A generalized ColeCole inversion scheme for SIP data.
    Args:
        *args: Arguments passed to the Inversion class.
        mesh: a two-dimensional mesh created with pygimli.meshtools
        frvec: (:obj:`list`) frequency vector
        data: data container created with pygimli.physics.ert.ertScheme
        n_cells: number of model cells : number of the mediums
        **kwargs: Additional keyword arguments passed to the Inversion class.
     """   
    def __init__(self, *args, mesh, frvec, data, n_cells=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_cells = n_cells
        # self.rhoaAllDatapoint = rhoaAllDatapoint
        # self.phiaAllDatapoint = phiaAllDatapoint
        # Add ColeCole parameters for each cell to dict
        range_n_cells = list(range(self.n_cells))
        # the length of the model vector is defind as follows:
        self.params.update({f'r0{i}': [0.9, 1.1]  for i in range_n_cells})
        self.params.update({f'm{i}': [0.0, 1.0] for i in range_n_cells})
        self.params.update({f'log_tau{i}': [-15, 5] for i in range_n_cells})
        self.params.update({f'c{i}': [0.0, 1.0] for i in range_n_cells})

        # self._bounds = np.array(self.param_bounds).T
        self.mesh = mesh 
        self.fdip = FDIP(f=frvec, data=data) # we need to initialize the frequency vecctor and synthetic data vector (data contaider)



    def forward(self, model, x=0):
        """Returns a ColeCole impedance.
        Args:
            mesh: 
            rhovec: ndarray of R0
            mvec: ndarray of m
            tauvec: ndarray of tau
            cvec: ndarray of c
        """
        mod = np.reshape(model, (4, -1))  # model is a long vector and we need to bring it to the shape
        AMP, PHI = self.fdip.simulate(self.mesh,
                              rhovec=mod[0],
                              mvec=mod[1],
                              tauvec=mod[2],
                              cvec=mod[3])
        print(AMP.shape, PHI.shape)
        return np.vstack((AMP.ravel(), PHI.ravel())) # take all the amplitude  and phase values and make a long data out of them, then combine together 
        # return self.fdip.simulate(self.mesh,
        #                           rhovec=model[0],
        #                           mvec=model[1],
        #                           tauvec=model[2],
        #                           cvec=model[3])

class fdipPeltonNew(Inversion):
    """A generalized ColeCole inversion scheme for SIP data.
    Args:
        *args: Arguments passed to the Inversion class.
        mesh: a two-dimensional mesh created with pygimli.meshtools
        frvec: (:obj:`list`) frequency vector
        data: data container created with pygimli.physics.ert.ertScheme
        n_cells: number of model cells : number of the mediums
        **kwargs: Additional keyword arguments passed to the Inversion class.
     """   
    def __init__(self, *args, mesh, frvec, data, n_cells=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_cells = n_cells
        # self.rhoaAllDatapoint = rhoaAllDatapoint
        # self.phiaAllDatapoint = phiaAllDatapoint
        # Add ColeCole parameters for each cell to dict
        range_n_cells = list(range(self.n_cells))
        # the length of the model vector is defind as follows:
        self.params.update({f'r0{i}': [0.9, 1.1]  for i in range_n_cells})
        self.params.update({f'm{i}': [0.0, 1.0] for i in range_n_cells})
        self.params.update({f'log_tau{i}': [-15, 5] for i in range_n_cells})
        self.params.update({f'c{i}': [0.0, 1.0] for i in range_n_cells})

        # self._bounds = np.array(self.param_bounds).T
        self.mesh = mesh 
        # self.fdip = FDIP(f=frvec, data=data) # we need to initialize the frequency vecctor and synthetic data vector (data contaider)
        self.freq = frvec
        self.fop = ert.ERTModelling(sr=True, verbose=False) # the forward operator is initialize only once
        self.fop.data = data
        self.fop.setMesh(mesh, ignoreRegionManager=True) # call mesh
        
    def forward(self, model, x=0):
        """Returns a ColeCole impedance.
        Args:
            mesh: 
            rhovec: ndarray of R0
            mvec: ndarray of m
            tauvec: ndarray of tau
            cvec: ndarray of c
        """
        mod = np.reshape(model, (4, -1))  # model is a long vector and we need to bring it to the shape
        AMP = np.zeros((self.fop.data.size(), len(self.freq))) # making a matrix for the amp
        PHI = np.zeros_like(AMP) # matrix for for phase
        
        for i, f in enumerate(self.freq):
            # compute Cole-Cole model
            res = modelColeColeRho(f, *mod) #compute complex resistivity as a function of frequency f
            resvec = res[self.mesh.cellMarkers()] # compute res for all number of cells in the mesh
            response = self.fop.response(resvec)
            # print(response)
            # print(response.shape)

            # sdfsfs
            AMP[:, i] = np.abs(response)
            phiai = -np.angle(response)
            phiai[phiai > np.pi/2] = np.pi - phiai[phiai > np.pi/2]
            PHI[:, i] = phiai

        return np.vstack((AMP.ravel(), PHI.ravel()))
    
    
    
    
    