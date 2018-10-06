from setuptools import setup

setup(
    name='cuda_gpupick',
    version='1.0.0',
    packages=['cuda_gpupick'],
    install_requires=[
        'numpy',
        'py3nvml'
    ],
    url='https://github.ibm.com/Alexis-Asseman/cuda-gpupick',
    license='MIT',
    author='Alexis Asseman',
    description='Automatically pick unused CUDA devices for your application, in a topology-aware fashion.',
    entry_points={
        'console_scripts': [
            'cuda-gpupick = cuda_gpupick.cuda_gpupick:main'
        ]
    }
)
