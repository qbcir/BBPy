from libcpp.memory cimport shared_ptr


cdef extern from "bb/MetricsBinaryAccuracy.h" namespace "bb":
    cdef cppclass _MetricsBinaryAccuracy "bb::MetricsBinaryAccuracy" [T]:
        @staticmethod
        shared_ptr[_MetricsBinaryAccuracy[T]] Create()


cdef class MetricsBinaryAccuracy:
    cdef shared_ptr[_MetricsBinaryAccuracy[float]] thisptr

    def __init__(self):
        self.thisptr = _MetricsBinaryAccuracy[float].Create()


cdef extern from "bb/MetricsCategoricalAccuracy.h" namespace "bb":
    cdef cppclass _MetricsCategoricalAccuracy "bb::MetricsCategoricalAccuracy" [T]:
        @staticmethod
        shared_ptr[_MetricsCategoricalAccuracy[T]] Create()


cdef class MetricsCategoricalAccuracy:
    cdef shared_ptr[_MetricsCategoricalAccuracy[float]] thisptr

    def __init__(self):
        self.thisptr = _MetricsCategoricalAccuracy[float].Create()


cdef extern from "bb/MetricsMeanSquaredError.h" namespace "bb":
    cdef cppclass _MetricsMeanSquaredError "bb::MetricsMeanSquaredError" [T]:
        @staticmethod
        shared_ptr[_MetricsMeanSquaredError[T]] Create()


cdef class MetricsMeanSquaredError:
    cdef shared_ptr[_MetricsMeanSquaredError[float]] thisptr

    def __init__(self):
        self.thisptr = _MetricsMeanSquaredError[float].Create()
