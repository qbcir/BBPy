import os
from os.path import join as pjoin
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


CUDA = locate_cuda()

CUDA_SRC = [
    'Manager.cu',
    'LocalHeap.cu',
    'FrameBufferCopy.cu',
    'Vector.cu',
    'MatrixColwiseSum.cu',
    'MatrixColwiseMeanVar.cu',
    'MatrixRowwiseSetVector.cu',
    'MicroMlp.cu',
    'BinaryLut6.cu',
    'SparseLut.cu',
    'StochasticLut.cu',
    'StochasticMaxPooling.cu',
    'StochasticBatchNormalization.cu',
    'ShuffleModulation.cu',
    'BinaryToReal.cu',
    'Im2Col.cu',
    'Col2Im.cu',
    'MaxPooling.cu',
    'UpSampling.cu',
    'BatchNormalization.cu',
    'ReLU.cu',
    'Sigmoid.cu',
    'Binarize.cu',
    'HardTanh.cu',
    'Adam.cu',
    'LossSoftmaxCrossEntropy.cu',
    'AccuracyCategoricalClassification.cu'
]

CUDA_SRC = [pjoin('BinaryBrain/cuda', src) for src in CUDA_SRC]

extensions = [
    Extension("BinaryBrain",
              language="c++",
              sources=["BBPy/BinaryBrain.pyx"] + CUDA_SRC,
              include_dirs=['BinaryBrain/include', CUDA['include']],
              library_dirs=[CUDA['lib64']],
              libraries=['omp', 'cudart', 'cublas'],
              extra_compile_args={
                  'gcc': ['-O2', '-mavx2', '-mfma', '-fopenmp', '-std=c++14', '-DBB_WITH_CUDA'],
                  'nvcc': [
                    '-gencode=arch=compute_35,code=sm_35',
                    '-gencode=arch=compute_75,code=sm_75',
                    '--compiler-options', "'-fPIC'"
                  ]
              })
]

setup(
    name='bbpy',
    ext_modules=cythonize(extensions, language_level="3"),
    cmdclass={'build_ext': custom_build_ext},
)
