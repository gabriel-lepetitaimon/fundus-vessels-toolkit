import setuptools
from torch.utils import cpp_extension

setuptools.setup(name='steered_kbase_cpp',
                 ext_modules=[cpp_extension.CppExtension('steered_kbase_cpp', ['steered_kbase_cpp.cpp'])],
                 cmdclass={'build_ext': cpp_extension.BuildExtension})
