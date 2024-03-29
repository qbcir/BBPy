ctypedef void (*callback_proc_t)(shared_ptr[_Model], void*)


cdef extern from "bb/Runner.h" namespace "bb":
    cdef cppclass _Runner "bb::Runner" [T]:
        @staticmethod
        shared_ptr[_Runner[T]] Create(
            string name,
            shared_ptr[_Model] net,
            index_t epoch_size,
            index_t batch_size,
            shared_ptr[_MetricsFunction] metrics_func,
            shared_ptr[_LossFunction] loss_func,
            shared_ptr[_Optimizer] optimizer,
            bool print_progress,
            bool file_read,
            bool file_write,
            bool write_serial,
            bool initial_evaluation,
            int64_t seed,
            callback_proc_t callback_proc,
            void* callback_user)

        void Fitting(_TrainData[T]& td, index_t epoch_size, index_t batch_size)
        double Evaluation(_TrainData[T]& td, index_t batch_size)


cdef class Runner:
    cdef shared_ptr[_Runner[float]] thisptr

    def __init__(self, name: str, net: Model, epoch_size: int, batch_size: int,
                 metrics_func: MetricsFunction, loss_func: LossFunction, optimizer: Optimizer,
                 print_progress=False, file_read=False, file_write=False, write_serial=False,
                 initial_evaluation=False, seed=1):
        cdef shared_ptr[_Model] _net = net.ptr()
        cdef shared_ptr[_MetricsFunction] _metrics_func = metrics_func.ptr()
        cdef shared_ptr[_LossFunction] _loss_func = loss_func.ptr()
        cdef shared_ptr[_Optimizer] _optimizer = optimizer.ptr()
        self.thisptr = _Runner[float].Create(
            name, _net, epoch_size, batch_size, _metrics_func, _loss_func, _optimizer,
            print_progress, file_read, file_write, write_serial, initial_evaluation, seed, NULL, NULL)

    def fit(self, train_data: TrainData, epoch_size: int, batch_size: int):
        deref(self.thisptr).Fitting(train_data._td, epoch_size, batch_size)

    def eval(self, train_data: TrainData, batch_size: int):
        return np.double(deref(self.thisptr).Evaluation(train_data._td, batch_size))
