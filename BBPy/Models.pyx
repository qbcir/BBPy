from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport intptr_t, uint64_t


cdef extern from "<iostream>":
    pass


ctypedef intptr_t index_t
ctypedef vector[index_t] indices_t

#ctypedef int LUT_SIZE_6 "6"


cdef extern from "bb/BatchNormalization.h" namespace "bb":
    cdef cppclass _BatchNormalization "bb::BatchNormalization" [T]:
        @staticmethod
        shared_ptr[_BatchNormalization[T]] Create(T momentum, T gamma, T beta)


cdef class BatchNormalization:
    cdef shared_ptr[_BatchNormalization[float]] thisptr

    def __init__(self, momentum, gamma, beta):
        self.thisptr = _BatchNormalization[float].Create(momentum, gamma, beta)


cdef extern from "bb/StochasticBatchNormalization.h" namespace "bb":
    cdef cppclass _StochasticBatchNormalization "bb::StochasticBatchNormalization" [T]:
        @staticmethod
        shared_ptr[_StochasticBatchNormalization[T]] Create(T momentum, T gamma, T beta)


cdef class StochasticBatchNormalization:
    cdef shared_ptr[_StochasticBatchNormalization[float]] thisptr

    def __init__(self, momentum, gamma, beta):
        self.thisptr = _StochasticBatchNormalization[float].Create(momentum, gamma, beta)


cdef extern from "bb/BackpropagatedBatchNormalization.h" namespace "bb":
    cdef cppclass _BackpropagatedBatchNormalization "bb::BackpropagatedBatchNormalization" [T]:
        @staticmethod
        shared_ptr[_BackpropagatedBatchNormalization[T]] Create(T gain, T beta)


cdef class BackpropagatedBatchNormalization:
    cdef shared_ptr[_BackpropagatedBatchNormalization[float]] thisptr

    def __init__(self, gain, beta):
        self.thisptr = _BackpropagatedBatchNormalization[float].Create(gain, beta)


cdef extern from "bb/ShuffleModulation.h" namespace "bb":
    cdef cppclass _ShuffleModulation "bb::ShuffleModulation<bb::Bit, float>":
        @staticmethod
        shared_ptr[_ShuffleModulation] Create(index_t shuffle_size, index_t lowering_size, uint64_t seed)


cdef class ShuffleModulation:
    cdef shared_ptr[_ShuffleModulation] thisptr

    def __init__(self, shuffle_size, lowering_size, seed):
        self.thisptr = _ShuffleModulation.Create(shuffle_size, lowering_size, seed)


cdef extern from "bb/Sequential.h" namespace "bb":
    cdef cppclass _Sequential "bb::Sequential":
        @staticmethod
        shared_ptr[_Sequential] Create()


cdef class Sequential:
    cdef shared_ptr[_Sequential] thisptr

    def __init__(self):
        self.thisptr = _Sequential.Create()


cdef extern from "bb/MaxPooling.h" namespace "bb":
    cdef cppclass _MaxPooling "bb::MaxPooling" [FT, BT]:
        @staticmethod
        shared_ptr[_MaxPooling[FT, BT]] Create(index_t filter_h_size, index_t filter_w_size)


cdef class MaxPooling:
    cdef shared_ptr[_MaxPooling[float, float]] thisptr

    def __init__(self, filter_h_size, filter_w_size):
        self.thisptr = _MaxPooling[float, float].Create(filter_h_size, filter_w_size)


cdef extern from "bb/Reduce.h" namespace "bb":
    cdef cppclass _Reduce "bb::Reduce" [FT, BT]:
        @staticmethod
        shared_ptr[_Reduce[FT, BT]] Create(indices_t output_shape)


cdef class Reduce:
    cdef shared_ptr[_Reduce[float, float]] thisptr

    def __init__(self, output_shape):
        self.thisptr = _Reduce[float, float].Create(output_shape)


cdef extern from "bb/ConcatenateCoefficient.h" namespace "bb":
    cdef cppclass _ConcatenateCoefficient "bb::ConcatenateCoefficient" [FT, BT]:
        @staticmethod
        shared_ptr[_ConcatenateCoefficient[FT, BT]] Create(index_t concatenate_size, uint64_t seed)


cdef class ConcatenateCoefficient:
    cdef shared_ptr[_ConcatenateCoefficient[float, float]] thisptr

    def __init__(self, concatenate_size, seed):
        self.thisptr = _ConcatenateCoefficient[float, float].Create(concatenate_size, seed)


cdef extern from "bb/ConvolutionCol2Im.h" namespace "bb":
    cdef cppclass _ConvolutionCol2Im "bb::ConvolutionCol2Im" [FT, BT]:
        @staticmethod
        shared_ptr[_ConvolutionCol2Im[FT, BT]] Create(index_t h_size, index_t w_size)


cdef class ConvolutionCol2Im:
    cdef shared_ptr[_ConvolutionCol2Im[float, float]] thisptr

    def __init__(self, h_size, w_size):
        self.thisptr = _ConvolutionCol2Im[float, float].Create(h_size, w_size)


cdef extern from "bb/ConvolutionIm2Col.h" namespace "bb":
    cdef cppclass _ConvolutionIm2Col "bb::ConvolutionIm2Col" [FT, BT]:
        @staticmethod
        shared_ptr[_ConvolutionIm2Col[FT, BT]] Create(index_t filter_h_size, index_t filter_w_size,
            index_t y_stride, index_t x_stride, string padding)


cdef class ConvolutionIm2Col:
    cdef shared_ptr[_ConvolutionIm2Col[float, float]] thisptr

    def __init__(self, filter_h_size, filter_w_size, y_stride, x_stride, padding):
        self.thisptr = _ConvolutionIm2Col[float, float].Create(filter_h_size, filter_w_size, y_stride, x_stride, padding)


cdef extern from "bb/BinaryLutN.h" namespace "bb":
    cdef cppclass _BinaryLut6 "bb::BinaryLutN<6, bb::Bit, float>":
        @staticmethod
        shared_ptr[_BinaryLut6] Create(const indices_t& output_shape, uint64_t seed)


cdef class BinaryLut6:
    cdef shared_ptr[_BinaryLut6] thisptr

    def __init__(self, output_shape, seed):
        self.thisptr = _BinaryLut6.Create(output_shape, seed)


cdef extern from "bb/SparseLutN.h" namespace "bb":
    cdef cppclass _SparseLut6 "bb::SparseLutN<6, bb::Bit, float>":
        @staticmethod
        shared_ptr[_SparseLut6] Create(const indices_t& output_shape, bool batch_norm, string connection, uint64_t seed)


cdef class SparseLut6:
    cdef shared_ptr[_SparseLut6] thisptr

    def __init__(self, output_shape, batch_norm, connection, seed):
        self.thisptr = _SparseLut6.Create(output_shape, batch_norm, connection, seed)


cdef extern from "bb/StochasticLutN.h" namespace "bb":
    cdef cppclass _StochasticLut6 "bb::StochasticLutN<6, float, float>":
        @staticmethod
        shared_ptr[_StochasticLut6] Create(const indices_t& output_shape, string connection, uint64_t seed)


cdef class StochasticLut6:
    cdef shared_ptr[_StochasticLut6] thisptr

    def __init__(self, output_shape, connection, seed):
        self.thisptr = _StochasticLut6.Create(output_shape, connection, seed)


#cdef extern from "bb/StochasticMaxPooling.h" namespace "bb":
#    cdef cppclass _StochasticMaxPooling "bb::StochasticMaxPooling" [FT, BT]:
#        @staticmethod
#        shared_ptr[_StochasticMaxPooling[FT, BT]] Create(index_t filter_h_size, index_t filter_w_size)


#cdef class StochasticMaxPooling:
#    cdef shared_ptr[_StochasticMaxPooling[float, float]] thisptr
#
#    def __init__(self, filter_h_size, filter_w_size):
#        self.thisptr = _StochasticMaxPooling[float, float].Create(filter_h_size, filter_w_size)