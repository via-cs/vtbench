from setuptools import setup, find_packages

setup(
    name='vtbench',
    version='0.1.0',
    description='VTBench: Visual Time-Series Benchmark for Classification',
    author='Madhumitha',
    author_email='mvenkat@ucdavis.com',
    url='https://github.com/yourusername/vtbench',  
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.9',
        'torchvision>=0.10',
        'numpy',
        'scikit-learn',
        'imbalanced-learn',
        'PyYAML',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'vtbench=vtbench.main:main',  
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',   
        'Operating System :: OS Independent',
    ],
)
