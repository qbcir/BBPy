from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("BinaryBrain",
              sources=["BBPy/BinaryBrain.pyx"],
              include_dirs=['BinaryBrain/include'],
              libraries=['omp'],
              extra_compile_args=['-O2', '-mavx2', '-mfma', '-fopenmp', '-std=c++14'],
              language="c++")
]

setup(
    name='bbpy',
    ext_modules=cythonize(extensions, language_level="3")
)
