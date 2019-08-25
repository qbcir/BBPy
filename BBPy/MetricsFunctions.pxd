from libcpp.memory cimport shared_ptr


cdef extern from "bb/MetricsFunction.h" namespace "bb":
    cdef cppclass _MetricsFunction "bb::MetricsFunction":
        pass


cdef class MetricsFunction:
    cdef shared_ptr[_MetricsFunction] ptr(self)