from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr, static_pointer_cast
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport intptr_t, int64_t, uint64_t
cimport numpy as np

import numpy as np


cdef extern from "<iostream>":
    pass

cdef extern from "<vector>":
    pass


ctypedef intptr_t index_t
ctypedef vector[index_t] indices_t

