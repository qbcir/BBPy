from libcpp.memory cimport shared_ptr


cdef extern from "bb/ReLU.h" namespace "bb":
    cdef cppclass _ReLU "bb::ReLU" [BT, RT]:
        @staticmethod
        shared_ptr[_ReLU[BT, RT]] Create()


cdef class ReLU:
    cdef shared_ptr[_ReLU[float, float]] thisptr

    def __init__(self):
        self.thisptr = _ReLU[float, float].Create()


cdef extern from "bb/Binarize.h" namespace "bb":
    cdef cppclass _Binarize "bb::Binarize" [BT, RT]:
        @staticmethod
        shared_ptr[_Binarize[BT, RT]] Create(RT binary_th, RT hardtanh_min, RT hardtanh_max)


cdef class Binarize:
    cdef shared_ptr[_Binarize[float, float]] thisptr

    def __init__(self, binary_th, hardtanh_min, hardtanh_max):
        self.thisptr = _Binarize[float, float].Create(binary_th, hardtanh_min, hardtanh_max)


cdef extern from "bb/Sigmoid.h" namespace "bb":
    cdef cppclass _Sigmoid "bb::Sigmoid" [T]:
        @staticmethod
        shared_ptr[_Sigmoid[T]] Create()


cdef class Sigmoid:
    cdef shared_ptr[_Sigmoid[float]] thisptr

    def __init__(self):
        self.thisptr = _Sigmoid[float].Create()


cdef extern from "bb/HardTanh.h" namespace "bb":
    cdef cppclass _HardTanh "bb::HardTanh" [BT, RT]:
        @staticmethod
        shared_ptr[_HardTanh[BT, RT]] Create(BT hardtanh_min, RT hardtanh_max)


cdef class HardTanh:
    cdef shared_ptr[_HardTanh[float, float]] thisptr

    def __init__(self, hardtanh_min, hardtanh_max):
        self.thisptr = _HardTanh[float, float].Create(hardtanh_min, hardtanh_max)
