from distutils.core import setup

setup(
    name='gpr',
    version='0.1.0',
    author='Dave Moore',
    author_email='dmoore@cs.berkeley.edu',
    packages=['gpr'],
    url='https://github.com/davmre/gpr',
    license='LICENSE',
    description='Gaussian Process Regression toolkit for Python/Numpy',
    long_description=open('README').read(),
    install_requires=[
        "numpy >= 1.1.0",
    ],
)

