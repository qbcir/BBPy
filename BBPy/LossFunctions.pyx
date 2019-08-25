from libcpp.memory cimport shared_ptr, static_pointer_cast

from LossFunctions cimport *


###############################################################################
cdef extern from "bb/LossMeanSquaredError.h" namespace "bb":
    cdef cppclass _LossMeanSquaredError "bb::LossMeanSquaredError" [T]:
        @staticmethod
        shared_ptr[_LossMeanSquaredError[T]] Create()


cdef extern from "bb/LossSoftmaxCrossEntropy.h" namespace "bb":
    cdef cppclass _LossSoftmaxCrossEntropy "bb::LossSoftmaxCrossEntropy" [T]:
        @staticmethod
        shared_ptr[_LossSoftmaxCrossEntropy[T]] Create()


###############################################################################
cdef class LossFunction:
    cdef shared_ptr[_LossFunction] ptr(self):
        cdef shared_ptr[_LossFunction] p
        return p


cdef class LossMeanSquaredError(LossFunction):
    cdef shared_ptr[_LossMeanSquaredError[float]] thisptr

    def __init__(self):
        self.thisptr = _LossMeanSquaredError[float].Create()

    cdef shared_ptr[_LossFunction] ptr(self):
        return static_pointer_cast[_LossFunction, _LossMeanSquaredError[float]](self.thisptr)


cdef class LossSoftmaxCrossEntropy(LossFunction):
    cdef shared_ptr[_LossSoftmaxCrossEntropy[float]] thisptr

    def __init__(self):
        self.thisptr = _LossSoftmaxCrossEntropy[float].Create()

    cdef shared_ptr[_LossFunction] ptr(self):
        return static_pointer_cast[_LossFunction, _LossSoftmaxCrossEntropy[float]](self.thisptr)