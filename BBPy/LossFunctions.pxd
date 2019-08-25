from libcpp.memory cimport shared_ptr


cdef extern from "bb/LossFunction.h" namespace "bb":
    cdef cppclass _LossFunction "bb::LossFunction":
        pass


cdef class LossFunction:
    cdef shared_ptr[_LossFunction] ptr(self)
