from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("*", sources=["BBPy/*.pyx"], include_dirs=['BinaryBrain/include'], language="c++")
]

setup(
    name='bbpy',
    ext_modules=cythonize(extensions, language_level="3")
)
