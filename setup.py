import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':
    setup(
        name='Pred2Plan',
        version=1.0,
        description='A closed-loop simulation & evaluation framework testing different SotA prediction models + downsized variants on different planners.',
        author='Mohamed-Khalil Bouzidi',
        author_email='mohamed-khalil.bouzidi@continental.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
    )
