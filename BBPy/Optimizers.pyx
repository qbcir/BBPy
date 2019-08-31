###############################################################################
cdef extern from "bb/Optimizer.h" namespace "bb":
    cdef cppclass _Optimizer "bb::Optimizer":
        pass

cdef extern from "bb/OptimizerAdam.h" namespace "bb":
    cdef cppclass _OptimizerAdam "bb::OptimizerAdam" [T]:
        @staticmethod
        shared_ptr[_OptimizerAdam[T]] Create(T learning_rate, T beta1, T beta2)


cdef extern from "bb/OptimizerAdaGrad.h" namespace "bb":
    cdef cppclass _OptimizerAdaGrad "bb::OptimizerAdaGrad" [T]:
        @staticmethod
        shared_ptr[_OptimizerAdaGrad[T]] Create(T learning_rate)


cdef extern from "bb/OptimizerSgd.h" namespace "bb":
    cdef cppclass _OptimizerSgd "bb::OptimizerSgd" [T]:
        @staticmethod
        shared_ptr[_OptimizerSgd[T]] Create(T learning_rate)


###############################################################################
cdef class Optimizer:
    cdef shared_ptr[_Optimizer] ptr(self):
        cdef shared_ptr[_Optimizer] p
        return p


cdef class OptimizerAdam(Optimizer):
    cdef shared_ptr[_OptimizerAdam[float]] thisptr

    def __init__(self, learning_rate, beta1, beta2):
        self.thisptr = _OptimizerAdam[float].Create(learning_rate, beta1, beta2)

    cdef shared_ptr[_Optimizer] ptr(self):
        return static_pointer_cast[_Optimizer, _OptimizerAdam[float]](self.thisptr)


cdef class OptimizerAdaGrad(Optimizer):
    cdef shared_ptr[_OptimizerAdaGrad[float]] thisptr

    def __init__(self, learning_rate):
        self.thisptr = _OptimizerAdaGrad[float].Create(learning_rate)

    cdef shared_ptr[_Optimizer] ptr(self):
        return static_pointer_cast[_Optimizer, _OptimizerAdaGrad[float]](self.thisptr)


cdef class OptimizerSgd(Optimizer):
    cdef shared_ptr[_OptimizerSgd[float]] thisptr

    def __init__(self, learning_rate):
        self.thisptr = _OptimizerSgd[float].Create(learning_rate)

    cdef shared_ptr[_Optimizer] ptr(self):
        return static_pointer_cast[_Optimizer, _OptimizerSgd[float]](self.thisptr)
