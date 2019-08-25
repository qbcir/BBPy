from libcpp.memory cimport shared_ptr


cdef extern from "<vector>":
    pass

cdef extern from "<iostream>":
    pass


cdef extern from "bb/Optimizer.h" namespace "bb":
    cdef cppclass _Optimizer "bb::Optimizer":
        pass


cdef class Optimizer:
    cdef shared_ptr[_Optimizer] ptr(self)