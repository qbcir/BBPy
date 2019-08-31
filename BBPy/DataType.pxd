from libcpp.vector cimport vector
from libc.stdint cimport intptr_t, uint64_t
from libcpp cimport bool


cdef extern from "<iostream>":
    pass

cdef extern from "<vector>":
    pass


ctypedef intptr_t index_t
ctypedef vector[index_t] indices_t


cdef extern from "bb/DataType.h" namespace "bb":
    cdef cppclass _Bit "bb::Bit":
        _Bit()
        _Bit (int v)
        _Bit(const _Bit& bit)
        _Bit(bool v)

        bool operator==(const _Bit &bit)
        bool operator>(const _Bit &bit)
        bool operator>=(const _Bit &bit)
        bool operator<(const _Bit &bit)
        bool operator<=(const _Bit &bit)


    cdef cppclass _TrainData "bb::TrainData" [T]:
        indices_t x_shape
        indices_t t_shape
        vector[vector[T]] x_train
        vector[vector[T]] t_train
        vector[vector[T]] x_test
        vector[vector[T]] t_test


cdef class TrainData:
    cdef _TrainData[float] _td

