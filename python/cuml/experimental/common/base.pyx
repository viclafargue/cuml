#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# distutils: language = c++

import typing
import functools

import numpy as np
import cupy as cp
import cuml
import cuml.common.logger as logger
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.input_utils import input_to_host_array
from cuml.common.array import CumlArray
from cuml.common.device_selection import DeviceType
from cuml.internals.api_decorators import device_interop_preparation
from cuml.internals import api_base_return_any, api_base_return_array, \
    api_base_return_generic, api_base_fit_transform, \
    api_base_return_any_skipall
from cuml.common.base import Base as originalBase


def dispatcher(func_name, gpu_func):
    @functools.wraps(gpu_func)
    def dispatch(self, *args, **kwargs):
        return self.dispatch_func(func_name, gpu_func, *args, **kwargs)
    return dispatch


def resolve_mro(cls, attr_name):
    for klass in cls.__mro__:
        if attr_name in vars(klass):
            attr = getattr(klass, attr_name)
            if not getattr(attr, '_cuml_dispatch_wrapped', False):
                return klass, attr
            else:
                return None
    return None


return_decorators = {
    'fit': api_base_return_any(),
    'predict': api_base_return_array(),
    'transform': api_base_return_array(),
    'kneighbors': api_base_return_generic(),
    'fit_transform': api_base_fit_transform(),
    'fit_predict': api_base_return_array(),
    'inverse_transform': api_base_return_array(),
    'score': api_base_return_any_skipall,
    'decision_function': api_base_return_array(),
    'predict_proba': api_base_return_array(),
    'predict_log_proba': api_base_return_array()
}


class Base(originalBase):
    """
    Experimental base class to implement CPU/GPU interoperability.
    """
    initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls.initialized:
            # add decorator to constructor to for interop feature
            cls.__init__ = device_interop_preparation(cls.__init__)

            estimator_methods = return_decorators.keys()
            for func_name in estimator_methods:
                # get cuML GPU function
                resolution = resolve_mro(cls, func_name)
                # if a function was found
                if resolution:
                    klass, gpu_func = resolution
                    new_func = return_decorators[func_name](dispatcher(func_name, gpu_func))
                    new_func._cuml_dispatch_wrapped = True
                    setattr(klass, func_name, new_func)

            # remember that class has been initialized
            cls.initialized = True

        # return an instance of the class
        return super().__new__(cls)


    def dispatch_func(self, func_name, gpu_func, *args, **kwargs):
        """
        This function will dispatch calls to training and inference according
        to the global configuration. It should work for all estimators
        sufficiently close the scikit-learn implementation as it uses
        it for training and inferences on host.

        Parameters
        ----------
        func_name : string
            name of the function to be dispatched
        gpu_func : function
            original cuML function
        args : arguments
            arguments to be passed to the function for the call
        kwargs : keyword arguments
            keyword arguments to be passed to the function for the call
        """
        # look for current device_type
        device_type = cuml.global_settings.device_type
        if device_type == DeviceType.device:
            # call the original cuml method
            return gpu_func(self, *args, **kwargs)
        elif device_type == DeviceType.host:
            # check if the sklean model already set as attribute of the cuml
            # estimator its presence should signify that CPU execution was
            # used previously
            if not hasattr(self, '_cpu_model'):
                filtered_kwargs = {}
                for keyword, arg in self._full_kwargs.items():
                    if keyword in self._cpu_hyperparams:
                        filtered_kwargs[keyword] = arg
                    else:
                        logger.info("Unused keyword parameter: {} "
                                    "during CPU estimator "
                                    "initialization".format(keyword))

                # initialize model
                self._cpu_model = self._cpu_model_class(**filtered_kwargs)

                # transfer attributes trained with cuml
                for attr in self.get_attr_names():
                    # check presence of attribute
                    if hasattr(self, attr) or \
                       isinstance(getattr(type(self), attr, None), property):
                        # get the cuml attribute
                        if hasattr(self, attr):
                            cu_attr = getattr(self, attr)
                        else:
                            cu_attr = getattr(type(self), attr).fget(self)
                        # if the cuml attribute is a CumlArrayDescriptorMeta
                        if hasattr(cu_attr, 'get_input_value'):
                            # extract the actual value from the
                            # CumlArrayDescriptorMeta
                            cu_attr_value = cu_attr.get_input_value()
                            # check if descriptor is empty
                            if cu_attr_value is not None:
                                if cu_attr.input_type == 'cuml':
                                    # transform cumlArray to numpy and set it
                                    # as an attribute in the CPU estimator
                                    setattr(self._cpu_model, attr,
                                            cu_attr_value.to_output('numpy'))
                                else:
                                    # transfer all other types of attributes
                                    # directly
                                    setattr(self._cpu_model, attr,
                                            cu_attr_value)
                        elif isinstance(cu_attr, CumlArray):
                            # transform cumlArray to numpy and set it
                            # as an attribute in the CPU estimator
                            setattr(self._cpu_model, attr,
                                    cu_attr.to_output('numpy'))
                        elif isinstance(cu_attr, cp.ndarray):
                            # transform cupy to numpy and set it
                            # as an attribute in the CPU estimator
                            setattr(self._cpu_model, attr,
                                    cp.asnumpy(cu_attr))
                        else:
                            # transfer all other types of attributes directly
                            setattr(self._cpu_model, attr, cu_attr)

            # converts all the args
            args = tuple(input_to_host_array(arg)[0] for arg in args)
            # converts all the kwarg
            for key, kwarg in kwargs.items():
                kwargs[key] = input_to_host_array(kwarg)[0]

            # call the method from the sklearn model
            cpu_func = getattr(self._cpu_model, func_name)
            res = cpu_func(*args, **kwargs)

            if func_name in ['fit', 'fit_transform', 'fit_predict']:
                # need to do this to mirror input type
                self._set_output_type(args[0])
                # always return the cuml estimator while training
                # mirror sk attributes to cuml after training
                for attr in self.get_attr_names():
                    # check presence of attribute
                    if hasattr(self._cpu_model, attr) or \
                       isinstance(getattr(type(self._cpu_model),
                                          attr, None), property):
                        # get the cpu attribute
                        if hasattr(self._cpu_model, attr):
                            cpu_attr = getattr(self._cpu_model, attr)
                        else:
                            cpu_attr = getattr(type(self._cpu_model),
                                               attr).fget(self._cpu_model)
                        # if the cpu attribute is an array
                        if isinstance(cpu_attr, np.ndarray):
                            # get data order wished for by CumlArrayDescriptor
                            if hasattr(self, attr + '_order'):
                                order = getattr(self, attr + '_order')
                            else:
                                order = 'K'
                            # transfer array to gpu and set it as a cuml
                            # attribute
                            cuml_array = input_to_cuml_array(cpu_attr,
                                                             order=order)[0]
                            setattr(self, attr, cuml_array)
                        else:
                            # transfer all other types of attributes directly
                            setattr(self, attr, cpu_attr)
                if func_name == 'fit':
                    return self
            # return method result
            return res
