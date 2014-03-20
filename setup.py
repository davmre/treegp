from distutils.core import setup
from distutils.core import setup, Extension
import numpy as np
import os

nonempty = lambda x: len(x) > 0

if os.getenv("C_INCLUDE_PATH") is not None:
    sys_includes = filter(nonempty, os.getenv("C_INCLUDE_PATH").split(':'))
else:
    sys_includes = []
if os.getenv("LIBRARY_PATH") is not None:
    sys_libraries = filter(nonempty, os.getenv("LIBRARY_PATH").split(':'))
else:
    sys_libraries = []

print sys_includes
print sys_libraries

#extra_compile_args = ['-g', '-pg']
extra_compile_args = ['-O3']

# extra_compile_args += [ '--stdlib=libc++'] # uncomment this for OSX/clang

#extra_link_args = ['-Wl,--strip-all']
#extra_link_args = ['-lrt',]
extra_link_args = []

ctree_root = 'src_c'
ctree_sources = ['cover_tree_point.cc', 'cover_tree_pp_debug.cc', 'distances.cc', 'vector_mult_py.cc', 'quadratic_form_py.cc']
from imp import find_module
f, pathname, descr = find_module("pyublas")
CTREE_INCLUDE_DIRS = [os.path.join(pathname, "include"),]

covertree_module = ctree = Extension('cover_tree',
                                     sources=[os.path.join(ctree_root, s) for s in ctree_sources],
                                     include_dirs=CTREE_INCLUDE_DIRS,
                                     library_dirs=['/'],
                                     libraries=['boost_python'],
                                     extra_compile_args=extra_compile_args,
                                     extra_link_args = extra_link_args,
                                 )
setup(
    name='treegp',
    version='0.1.0',
    author='Dave Moore',
    author_email='dmoore@cs.berkeley.edu',
    packages=['treegp'],
    url='https://github.com/davmre/treegp',
    license='LICENSE',
    description='Gaussian Process Regression toolkit for Python/Numpy',
    long_description=open('README').read(),
    install_requires=[
        "numpy >= 1.1.0",
    ],
    include_dirs=[np.get_include()] + sys_includes,
    ext_modules=[covertree_module,]
)
