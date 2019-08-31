###############################################################################
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

    def __init__(self, x_train: np.ndarray, t_train: np.ndarray, x_test: np.ndarray, t_test: np.ndarray):
        #
        for dim in x_train.shape[:-1]:
            self._td.x_shape.push_back(dim)
        for dim in t_train.shape[:-1]:
            self._td.t_shape.push_back(dim)
        #
        x_train_ = x_train.reshape(-1, x_train.shape[-1])
        t_train_ = t_train.reshape(-1, t_train.shape[-1])
        x_test_ = x_test.reshape(-1, x_test.shape[-1])
        t_test_ = t_test.reshape(-1, t_test.shape[-1])
        #
        self._td.x_train.resize(x_train.shape[-1])
        for i in range(x_train.shape[-1]):
            self._td.x_train[i].resize(x_train_.shape[0])
            for j in range(x_train_.shape[0]):
                self._dt.x_train[i][j] = x_train_[j, i]
        #
        self._td.t_train.resize(t_train.shape[-1])
        for i in range(t_train.shape[-1]):
            self._td.t_train[i].resize(t_train_.shape[0])
            for j in range(t_train_.shape[0]):
                self._dt.t_train[i][j] = t_train_[j, i]
        #
        self._td.x_test.resize(x_test.shape[-1])
        for i in range(x_test.shape[-1]):
            self._td.x_test[i].resize(x_test_.shape[0])
            for j in range(x_test_.shape[0]):
                self._dt.x_test[i][j] = x_test_[j, i]
        #
        self._td.t_test.resize(t_test.shape[-1])
        for i in range(t_test.shape[-1]):
            self._td.t_test[i].resize(t_test_.shape[0])
            for j in range(t_test_.shape[0]):
                self._dt.t_test[i][j] = t_test_[j, i]
