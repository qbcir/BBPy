from libcpp.memory cimport shared_ptr


cdef extern from "bb/Sequential.h" namespace "bb":
    cdef cppclass _Sequential "bb::Sequential":
        @staticmethod
        shared_ptr[_Sequential] Create()


cdef class Sequential:
    cdef shared_ptr[_Sequential] thisptr

    def __init__(self):
        self.thisptr = _Sequential.Create()
