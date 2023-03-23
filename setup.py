from setuptools import setup

setup(
    name='thesis',
    install_requires=[
        'pip',
        'torch',
        'torchvision',
        'black',
        'tensorboardX',
        'opencv-python',
        'torch-tb-profiler',
        'nvidia-pyindex',
        'nvidia-dali-cuda110',
        'cupy-cuda11x',
        'multiprocess',
        'matplotlib'
    ],
    dependency_links=[
    ]
)
