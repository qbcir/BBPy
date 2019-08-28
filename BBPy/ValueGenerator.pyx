from libcpp.memory cimport shared_ptr, static_pointer_cast
from libc.stdint cimport int64_t
from libcpp cimport bool

from DataType cimport *
from ValueGenerator cimport *


###############################################################################
cdef extern from "bb/NormalDistributionGenerator.h" namespace "bb":
    cdef cppclass _NormalDistributionGenerator "bb::NormalDistributionGenerator" [T]:
        @staticmethod
        shared_ptr[_NormalDistributionGenerator[T]] Create(T mean, T stddev, int64_t seed)


cdef extern from "bb/UniformDistributionGenerator.h" namespace "bb":
    cdef cppclass _UniformDistributionGenerator "bb::UniformDistributionGenerator" [T]:
        @staticmethod
        shared_ptr[_UniformDistributionGenerator[T]] Create(T a, T b, int64_t seed)

    cdef cppclass _UniformDistributionBitGenerator "bb::UniformDistributionGenerator<bb::Bit>":
        @staticmethod
        shared_ptr[_UniformDistributionBitGenerator] Create(_Bit a, _Bit b, int64_t seed)


###############################################################################
cdef class FloatValueGenerator:
    cdef shared_ptr[_FloatValueGenerator] ptr(self):
        cdef shared_ptr[_FloatValueGenerator] p
        return p


cdef class NormalDistributionGenerator(FloatValueGenerator):
    cdef shared_ptr[_NormalDistributionGenerator[float]] thisptr

    def __init__(self, mean: float = 0.0, stddev: float = 1.0, seed: int = 1):
        self.thisptr = _NormalDistributionGenerator[float].Create(mean, stddev, seed)

    cdef shared_ptr[_FloatValueGenerator] ptr(self):
        return static_pointer_cast[_FloatValueGenerator, _NormalDistributionGenerator[float]](self.thisptr)


cdef class UniformDistributionGenerator(FloatValueGenerator):
    cdef shared_ptr[_UniformDistributionGenerator[float]] thisptr

    def __init__(self, a: float = 0.0, b: float = 1.0, seed: int = 1):
        self.thisptr = _UniformDistributionGenerator[float].Create(a, b, seed)

    cdef shared_ptr[_FloatValueGenerator] ptr(self):
        return static_pointer_cast[_FloatValueGenerator, _UniformDistributionGenerator[float]](self.thisptr)


cdef class BitValueGenerator:
    cdef shared_ptr[_BitValueGenerator] ptr(self):
        cdef shared_ptr[_BitValueGenerator] p
        return p


cdef class UniformDistributionBitGenerator(BitValueGenerator):
    cdef shared_ptr[_UniformDistributionBitGenerator] thisptr

    def __init__(self, a: int = 0, b: int = 1, seed: int = 1):
        cdef int _a = a
        cdef int _b = b
        self.thisptr = _UniformDistributionBitGenerator.Create(_Bit(_a), _Bit(_b), seed)

    cdef shared_ptr[_BitValueGenerator] ptr(self):
        return static_pointer_cast[_BitValueGenerator, _UniformDistributionBitGenerator](self.thisptr)
