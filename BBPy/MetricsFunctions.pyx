from libcpp.memory cimport shared_ptr, static_pointer_cast

from MetricsFunctions cimport *


###############################################################################
cdef extern from "bb/MetricsBinaryAccuracy.h" namespace "bb":
    cdef cppclass _MetricsBinaryAccuracy "bb::MetricsBinaryAccuracy" [T]:
        @staticmethod
        shared_ptr[_MetricsBinaryAccuracy[T]] Create()


cdef extern from "bb/MetricsCategoricalAccuracy.h" namespace "bb":
    cdef cppclass _MetricsCategoricalAccuracy "bb::MetricsCategoricalAccuracy" [T]:
        @staticmethod
        shared_ptr[_MetricsCategoricalAccuracy[T]] Create()


cdef extern from "bb/MetricsMeanSquaredError.h" namespace "bb":
    cdef cppclass _MetricsMeanSquaredError "bb::MetricsMeanSquaredError" [T]:
        @staticmethod
        shared_ptr[_MetricsMeanSquaredError[T]] Create()


###############################################################################
cdef class MetricsFunction:
    cdef shared_ptr[_MetricsFunction] ptr(self):
        cdef shared_ptr[_MetricsFunction] p
        return p


cdef class MetricsBinaryAccuracy(MetricsFunction):
    cdef shared_ptr[_MetricsBinaryAccuracy[float]] thisptr

    def __init__(self):
        self.thisptr = _MetricsBinaryAccuracy[float].Create()

    cdef shared_ptr[_MetricsFunction] ptr(self):
        return static_pointer_cast[_MetricsFunction, _MetricsBinaryAccuracy[float]](self.thisptr)


cdef class MetricsCategoricalAccuracy(MetricsFunction):
    cdef shared_ptr[_MetricsCategoricalAccuracy[float]] thisptr

    def __init__(self):
        self.thisptr = _MetricsCategoricalAccuracy[float].Create()

    cdef shared_ptr[_MetricsFunction] ptr(self):
        return static_pointer_cast[_MetricsFunction, _MetricsCategoricalAccuracy[float]](self.thisptr)


cdef class MetricsMeanSquaredError(MetricsFunction):
    cdef shared_ptr[_MetricsMeanSquaredError[float]] thisptr

    def __init__(self):
        self.thisptr = _MetricsMeanSquaredError[float].Create()

    cdef shared_ptr[_MetricsFunction] ptr(self):
        return static_pointer_cast[_MetricsFunction, _MetricsMeanSquaredError[float]](self.thisptr)
