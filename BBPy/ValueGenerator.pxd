from libcpp.memory cimport shared_ptr


cdef extern from "bb/ValueGenerator.h" namespace "bb":
    cdef cppclass _FloatValueGenerator "bb::ValueGenerator<float>":
        pass

    cdef cppclass _BitValueGenerator "bb::ValueGenerator<bb::Bit>":
        pass


cdef class FloatValueGenerator:
    cdef shared_ptr[_FloatValueGenerator] ptr(self)


cdef class BitValueGenerator:
    cdef shared_ptr[_BitValueGenerator] ptr(self)
