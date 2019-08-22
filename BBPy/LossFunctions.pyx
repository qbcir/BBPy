from libcpp.memory cimport shared_ptr


cdef extern from "bb/LossMeanSquaredError.h" namespace "bb":
    cdef cppclass _LossMeanSquaredError "bb::LossMeanSquaredError" [T]:
        @staticmethod
        shared_ptr[_LossMeanSquaredError[T]] Create()


cdef class LossMeanSquaredError:
    cdef shared_ptr[_LossMeanSquaredError[float]] thisptr

    def __init__(self):
        self.thisptr = _LossMeanSquaredError[float].Create()


cdef extern from "bb/LossSoftmaxCrossEntropy.h" namespace "bb":
    cdef cppclass _LossSoftmaxCrossEntropy "bb::LossSoftmaxCrossEntropy" [T]:
        @staticmethod
        shared_ptr[_LossSoftmaxCrossEntropy[T]] Create()


cdef class LossSoftmaxCrossEntropy:
    cdef shared_ptr[_LossSoftmaxCrossEntropy[float]] thisptr

    def __init__(self):
        self.thisptr = _LossSoftmaxCrossEntropy[float].Create()