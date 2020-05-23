import setuptools

with open('requirements.txt', 'r') as fc:
    requirements = [line.strip() for line in fc]

setuptools.setup(
    name='barbellcv',
    version='0.0.0',
    author='Trevor Lancon',
    description='Interactive velocity-based tracking for barbell movements.',
    long_description='Track the path, velocity, power output, and time to complete reps for barbell movements using'
                     'only a laptop and a webcam.',
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>3.6',
    install_requires=requirements
)
