from setuptools import setup, find_packages

__version__ = "2.0.0"

setup(
    name='sticky_pi_ml',
    version=__version__,
    long_description=__doc__,
    packages=find_packages(),
    scripts=['bin/universal_insect_detector.py',
             'bin/siamese_insect_matcher.py',
             'bin/insect_tuboid_classifier.py',
             'bin/standalone_uid.py',
             'bin/standalone_sim.py',
             'bin/standalone_itc.py',

             ],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'python-dotenv',
        'numpy',
        'pandas',
        'ffmpeg-python',
        'svgpathtools',
        'CairoSVG',
        'opencv_python',
        'networkx',
        'detectron2',
        'torch >= 1.4',
        'shapely',
        'torchvision',
        'sklearn'],
    extras_require={
        'client': ['sticky_pi_api'],
        'test': ['nose', 'pytest', 'pytest-cov', 'codecov', 'coverage'],
        'docs': ['mock', 'sphinx-autodoc-typehints', 'sphinx', 'sphinx_rtd_theme', 'recommonmark', 'mock']
    },
    test_suite='nose.collector'
)

exec(open('sticky_pi_ml/_version.py').read())

