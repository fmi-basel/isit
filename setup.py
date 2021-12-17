from setuptools import setup, find_packages

contrib = [
    'Raphael Ortiz',
]

# setup.
setup(name='isit',
      version='0.1.3',
      description='2D/3D instance segmentation with iterative training',
      author=', '.join(contrib),
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'scikit-image',
          'imagecodecs',
          'pytest',
          'luigi',
          'dl-utils @ git+https://github.com/fmi-basel/dl-utils',
          'improc @ git+https://github.com/fmi-basel/improc',
          'inter_view @ git+https://github.com/fmi-basel/inter-view',
      ],
      zip_safe=False)
