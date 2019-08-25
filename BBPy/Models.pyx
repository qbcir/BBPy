from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr, static_pointer_cast

from Models cimport *


###############################################################################
cdef extern from "bb/BatchNormalization.h" namespace "bb":
    cdef cppclass _BatchNormalization "bb::BatchNormalization" [T]:
        @staticmethod
        shared_ptr[_BatchNormalization[T]] Create(T momentum, T gamma, T beta)


cdef extern from "bb/StochasticBatchNormalization.h" namespace "bb":
    cdef cppclass _StochasticBatchNormalization "bb::StochasticBatchNormalization" [T]:
        @staticmethod
        shared_ptr[_StochasticBatchNormalization[T]] Create(T momentum, T gamma, T beta)


cdef extern from "bb/BackpropagatedBatchNormalization.h" namespace "bb":
    cdef cppclass _BackpropagatedBatchNormalization "bb::BackpropagatedBatchNormalization" [T]:
        @staticmethod
        shared_ptr[_BackpropagatedBatchNormalization[T]] Create(T gain, T beta)


cdef extern from "bb/ShuffleModulation.h" namespace "bb":
    cdef cppclass _ShuffleModulation "bb::ShuffleModulation<bb::Bit, float>":
        @staticmethod
        shared_ptr[_ShuffleModulation] Create(index_t shuffle_size, index_t lowering_size, uint64_t seed)


cdef extern from "bb/DenseAffine.h" namespace "bb":
    cdef cppclass _DenseAffine "bb::DenseAffine" [T]:
        @staticmethod
        shared_ptr[_DenseAffine[T]] Create(const indices_t& output_shape)


cdef extern from "bb/Sequential.h" namespace "bb":
    cdef cppclass _Sequential "bb::Sequential":
        @staticmethod
        shared_ptr[_Sequential] Create()

        void Add(shared_ptr[_Model] layer)


cdef extern from "bb/MaxPooling.h" namespace "bb":
    cdef cppclass _MaxPooling "bb::MaxPooling" [FT, BT]:
        @staticmethod
        shared_ptr[_MaxPooling[FT, BT]] Create(index_t filter_h_size, index_t filter_w_size)


cdef extern from "bb/Reduce.h" namespace "bb":
    cdef cppclass _Reduce "bb::Reduce" [FT, BT]:
        @staticmethod
        shared_ptr[_Reduce[FT, BT]] Create(indices_t output_shape)


cdef extern from "bb/BinaryToReal.h" namespace "bb":
    cdef cppclass _BinaryToReal "bb::BinaryToReal" [FT, BT]:
        @staticmethod
        shared_ptr[_BinaryToReal[FT, BT]] Create(index_t modulation_size, indices_t output_shape)


cdef extern from "bb/LoweringConvolution.h" namespace "bb":
    cdef cppclass _LoweringConvolution "bb::LoweringConvolution" [FT, BT]:
        @staticmethod
        shared_ptr[_LoweringConvolution[FT, BT]] Create(shared_ptr[_Model] layer,
            index_t filter_h_size, index_t filter_w_size, index_t y_stride, index_t x_stride, string padding)


cdef extern from "bb/UpSampling.h" namespace "bb":
    cdef cppclass _UpSampling "bb::UpSampling" [FT, BT]:
        @staticmethod
        shared_ptr[_UpSampling[FT, BT]] Create(index_t filter_h_size, index_t filter_w_size, bool fill)


cdef extern from "bb/ConcatenateCoefficient.h" namespace "bb":
    cdef cppclass _ConcatenateCoefficient "bb::ConcatenateCoefficient" [FT, BT]:
        @staticmethod
        shared_ptr[_ConcatenateCoefficient[FT, BT]] Create(index_t concatenate_size, uint64_t seed)


cdef extern from "bb/ConcatenateCoefficient.h" namespace "bb":
    cdef cppclass _ConcatenateCoefficient "bb::ConcatenateCoefficient" [FT, BT]:
        @staticmethod
        shared_ptr[_ConcatenateCoefficient[FT, BT]] Create(index_t concatenate_size, uint64_t seed)


cdef extern from "bb/ConvolutionCol2Im.h" namespace "bb":
    cdef cppclass _ConvolutionCol2Im "bb::ConvolutionCol2Im" [FT, BT]:
        @staticmethod
        shared_ptr[_ConvolutionCol2Im[FT, BT]] Create(index_t h_size, index_t w_size)


cdef extern from "bb/ConvolutionIm2Col.h" namespace "bb":
    cdef cppclass _ConvolutionIm2Col "bb::ConvolutionIm2Col" [FT, BT]:
        @staticmethod
        shared_ptr[_ConvolutionIm2Col[FT, BT]] Create(index_t filter_h_size, index_t filter_w_size,
            index_t y_stride, index_t x_stride, string padding)


cdef extern from "bb/BinaryLutN.h" namespace "bb":
    cdef cppclass _BinaryLut6 "bb::BinaryLutN<6, bb::Bit, float>":
        @staticmethod
        shared_ptr[_BinaryLut6] Create(const indices_t& output_shape, uint64_t seed)


cdef extern from "bb/SparseLutN.h" namespace "bb":
    cdef cppclass _SparseLut6 "bb::SparseLutN<6, bb::Bit, float>":
        @staticmethod
        shared_ptr[_SparseLut6] Create(const indices_t& output_shape, bool batch_norm, string connection, uint64_t seed)


cdef extern from "bb/SparseLutDiscreteN.h" namespace "bb":
    cdef cppclass _SparseLutDiscrete6 "bb::SparseLutDiscreteN<6, bb::Bit, float>":
        @staticmethod
        shared_ptr[_SparseLutDiscrete6] Create(
            const indices_t& output_shape,
            bool batch_norm,
            string connection,
            uint64_t seed)


cdef extern from "bb/StochasticLutN.h" namespace "bb":
    cdef cppclass _StochasticLut6 "bb::StochasticLutN<6, float, float>":
        @staticmethod
        shared_ptr[_StochasticLut6] Create(const indices_t& output_shape, string connection, uint64_t seed)

###############################################################################
cdef class Model:
    cdef shared_ptr[_Model] ptr(self):
        cdef shared_ptr[_Model] p
        return p


cdef class BatchNormalization(Model):
    cdef shared_ptr[_BatchNormalization[float]] thisptr

    def __init__(self, momentum, gamma, beta):
        self.thisptr = _BatchNormalization[float].Create(momentum, gamma, beta)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _BatchNormalization[float]](self.thisptr)


cdef class StochasticBatchNormalization(Model):
    cdef shared_ptr[_StochasticBatchNormalization[float]] thisptr

    def __init__(self, momentum, gamma, beta):
        self.thisptr = _StochasticBatchNormalization[float].Create(momentum, gamma, beta)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _StochasticBatchNormalization[float]](self.thisptr)


cdef class BackpropagatedBatchNormalization(Model):
    cdef shared_ptr[_BackpropagatedBatchNormalization[float]] thisptr

    def __init__(self, gain, beta):
        self.thisptr = _BackpropagatedBatchNormalization[float].Create(gain, beta)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _BackpropagatedBatchNormalization[float]](self.thisptr)


cdef class ShuffleModulation(Model):
    cdef shared_ptr[_ShuffleModulation] thisptr

    def __init__(self, shuffle_size, lowering_size, seed):
        self.thisptr = _ShuffleModulation.Create(shuffle_size, lowering_size, seed)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _ShuffleModulation](self.thisptr)


cdef class DenseAffine(Model):
    cdef shared_ptr[_DenseAffine[float]] thisptr

    def __init__(self, output_shape):
        self.thisptr = _DenseAffine[float].Create(output_shape)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _DenseAffine[float]](self.thisptr)


cdef class Sequential(Model):
    cdef shared_ptr[_Sequential] thisptr

    def __init__(self):
        self.thisptr = _Sequential.Create()

    def Add(self, model: Model):
        cdef shared_ptr[_Model] _model = model.ptr()
        deref(self.thisptr).Add(_model)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _Sequential](self.thisptr)


cdef class MaxPooling(Model):
    cdef shared_ptr[_MaxPooling[float, float]] thisptr

    def __init__(self, filter_h_size, filter_w_size):
        self.thisptr = _MaxPooling[float, float].Create(filter_h_size, filter_w_size)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _MaxPooling[float, float]](self.thisptr)


cdef class Reduce(Model):
    cdef shared_ptr[_Reduce[float, float]] thisptr

    def __init__(self, output_shape):
        self.thisptr = _Reduce[float, float].Create(output_shape)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _Reduce[float, float]](self.thisptr)


cdef class BinaryToReal(Model):
    cdef shared_ptr[_BinaryToReal[float, float]] thisptr

    def __init__(self, modulation_size, output_shape):
        self.thisptr = _BinaryToReal[float, float].Create(modulation_size, output_shape)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _BinaryToReal[float, float]](self.thisptr)


cdef class UpSampling(Model):
    cdef shared_ptr[_UpSampling[float, float]] thisptr

    def __init__(self, filter_h_size, filter_w_size, fill):
        self.thisptr = _UpSampling[float, float].Create(filter_h_size, filter_w_size, fill)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _UpSampling[float, float]](self.thisptr)


cdef class ConcatenateCoefficient(Model):
    cdef shared_ptr[_ConcatenateCoefficient[float, float]] thisptr

    def __init__(self, concatenate_size, seed):
        self.thisptr = _ConcatenateCoefficient[float, float].Create(concatenate_size, seed)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _ConcatenateCoefficient[float, float]](self.thisptr)


cdef class ConvolutionCol2Im(Model):
    cdef shared_ptr[_ConvolutionCol2Im[float, float]] thisptr

    def __init__(self, h_size, w_size):
        self.thisptr = _ConvolutionCol2Im[float, float].Create(h_size, w_size)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _ConvolutionCol2Im[float, float]](self.thisptr)


cdef class ConvolutionIm2Col(Model):
    cdef shared_ptr[_ConvolutionIm2Col[float, float]] thisptr

    def __init__(self, filter_h_size, filter_w_size, y_stride, x_stride, padding):
        self.thisptr = _ConvolutionIm2Col[float, float].Create(filter_h_size, filter_w_size, y_stride, x_stride, padding)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _ConvolutionIm2Col[float, float]](self.thisptr)



cdef class LoweringConvolution(Model):
    cdef shared_ptr[_LoweringConvolution[float, float]] thisptr

    def __init__(self, layer: Model, filter_h_size: int, filter_w_size: int,
                 y_stride=1, x_stride=1, padding="valid"):
        cdef shared_ptr[_Model] _layer = layer.ptr()
        self.thisptr = _LoweringConvolution[float, float].Create(_layer, filter_h_size, filter_w_size, y_stride, x_stride, padding)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _LoweringConvolution[float, float]](self.thisptr)


# FIXME
#cdef extern from "bb/AveragePooling.h" namespace "bb":
#    cdef cppclass _AveragePooling "bb::AveragePooling" [FT, BT]:
#        @staticmethod
#        shared_ptr[_AveragePooling[FT, BT]] Create(index_t filter_h_size, index_t filter_w_size)


#cdef class AveragePooling:
#    cdef shared_ptr[_AveragePooling[float, float]] thisptr
#
#    def __init__(self, filter_h_size, filter_w_size):
#        self.thisptr = _AveragePooling[float, float].Create(filter_h_size, filter_w_size)


cdef class BinaryLut6(Model):
    cdef shared_ptr[_BinaryLut6] thisptr

    def __init__(self, output_shape, seed):
        self.thisptr = _BinaryLut6.Create(output_shape, seed)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _BinaryLut6](self.thisptr)


cdef class SparseLut6(Model):
    cdef shared_ptr[_SparseLut6] thisptr

    def __init__(self, output_shape, batch_norm, connection, seed):
        self.thisptr = _SparseLut6.Create(output_shape, batch_norm, connection, seed)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _SparseLut6](self.thisptr)


cdef class SparseLutDiscrete6(Model):
    cdef shared_ptr[_SparseLutDiscrete6] thisptr

    def __init__(self, output_shape, batch_norm, connection, seed):
        self.thisptr = _SparseLutDiscrete6.Create(output_shape, batch_norm, connection, seed)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _SparseLutDiscrete6](self.thisptr)


# FIXME
#cdef extern from "bb/SparseBinaryLutN.h" namespace "bb":
#    cdef cppclass _SparseBinaryLut6 "bb::SparseBinaryLutN<6, bb::Bit, float>":
#        @staticmethod
#        shared_ptr[_SparseBinaryLut6] Create(const indices_t& output_shape, string connection, uint64_t seed)
#
#
#cdef class SparseBinaryLut6:
#    cdef shared_ptr[_SparseBinaryLut6] thisptr
#
#    def __init__(self, output_shape, connection, seed):
#        self.thisptr = _SparseBinaryLut6.Create(output_shape, connection, seed)


cdef class StochasticLut6(Model):
    cdef shared_ptr[_StochasticLut6] thisptr

    def __init__(self, output_shape, connection, seed):
        self.thisptr = _StochasticLut6.Create(output_shape, connection, seed)

    cdef shared_ptr[_Model] ptr(self):
        return static_pointer_cast[_Model, _StochasticLut6](self.thisptr)

#FIXME
#cdef extern from "bb/StochasticMaxPooling.h" namespace "bb":
#    cdef cppclass _StochasticMaxPooling "bb::StochasticMaxPooling" [FT, BT]:
#        @staticmethod
#        shared_ptr[_StochasticMaxPooling[FT, BT]] Create(index_t filter_h_size, index_t filter_w_size)


#cdef class StochasticMaxPooling:
#    cdef shared_ptr[_StochasticMaxPooling[float, float]] thisptr
#
#    def __init__(self, filter_h_size, filter_w_size):
#        self.thisptr = _StochasticMaxPooling[float, float].Create(filter_h_size, filter_w_size)