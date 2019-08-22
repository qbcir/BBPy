from libcpp.memory cimport shared_ptr


cdef extern from "<vector>":
    pass

cdef extern from "<iostream>":
    pass

cdef extern from "bb/OptimizerAdam.h" namespace "bb":
    cdef cppclass _OptimizerAdam "bb::OptimizerAdam" [T]:
        @staticmethod
        shared_ptr[_OptimizerAdam[T]] Create(T learning_rate, T beta1, T beta2)


cdef class OptimizerAdam:
    cdef shared_ptr[_OptimizerAdam[float]] thisptr

    def __init__(self, learning_rate, beta1, beta2):
        self.thisptr = _OptimizerAdam[float].Create(learning_rate, beta1, beta2)


cdef extern from "bb/OptimizerAdaGrad.h" namespace "bb":
    cdef cppclass _OptimizerAdaGrad "bb::OptimizerAdaGrad" [T]:
        @staticmethod
        shared_ptr[_OptimizerAdaGrad[T]] Create(T learning_rate)


cdef class OptimizerAdaGrad:
    cdef shared_ptr[_OptimizerAdaGrad[float]] thisptr

    def __init__(self, learning_rate):
        self.thisptr = _OptimizerAdaGrad[float].Create(learning_rate)


cdef extern from "bb/OptimizerSgd.h" namespace "bb":
    cdef cppclass _OptimizerSgd "bb::OptimizerSgd" [T]:
        @staticmethod
        shared_ptr[_OptimizerSgd[T]] Create(T learning_rate)


cdef class OptimizerSgd:
    cdef shared_ptr[_OptimizerSgd[float]] thisptr

    def __init__(self, learning_rate):
        self.thisptr = _OptimizerSgd[float].Create(learning_rate)
