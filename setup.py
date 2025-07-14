from setuptools import setup, find_packages

contrib = [
    'Raphael Ortiz',
    "Franziska Moos",
    "Enrico Tagliavini"
]

# setup.
setup(name='isit',
      version='0.1.8',
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
          'tensorflow<2.8',
          'dl-utils @ git+https://github.com/fmi-basel/dl-utils@dev-enrico',
          'improc @ git+https://github.com/fmi-basel/improc@dev-franzi',
          #'inter_view @ git+https://github.com/fmi-basel/inter-view',
      ],
      extras_require={
          "fractal-tasks-core": ["fractal-tasks-core==1.2.2", "pydantic<2.8"],
          },
      zip_safe=False,
      data_files = [("", ["isit/__FRACTAL_MANIFEST__.json"])]
      )
