from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport intptr_t, uint64_t


cdef extern from "bb/Model.h" namespace "bb":
    cdef cppclass _Model "bb::Model":
        pass


cdef class Model:
    cdef shared_ptr[_Model] ptr(self)
