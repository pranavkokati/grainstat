from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Professional grain size analysis for materials science"

# Read requirements
def read_requirements():
    try:
        with open('requirements.txt', 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return [
            'numpy>=1.20.0',
            'scipy>=1.7.0',
            'scikit-image>=0.18.0',
            'matplotlib>=3.3.0',
            'pandas>=1.3.0',
            'Pillow>=8.0.0',
            'seaborn>=0.11.0'
        ]

setup(
    name='grainstat',
    version='1.0.0',
    author='Materials Science Lab',
    author_email='contact@materialslab.com',
    description='Professional grain size analysis for materials science',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/materialslab/grainstat',
    packages=find_packages(where='.', include=['grainstat*']),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.900',
        ],
        'pdf': [
            'reportlab>=3.5.0',
        ],
        'interactive': [
            'ipywidgets>=7.0',
            'jupyter>=1.0',
        ],
        'all': [
            'reportlab>=3.5.0',
            'ipywidgets>=7.0',
            'jupyter>=1.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'grainstat=grainstat.cli:cli_main',
        ],
    },
    package_data={
        'grainstat': [
            'export/templates/*.html',
            'export/templates/*.md',
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'materials science',
        'grain analysis',
        'microstructure',
        'image processing',
        'metallurgy',
        'microscopy',
        'SEM',
        'optical microscopy',
        'ASTM',
        'grain size'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/materialslab/grainstat/issues',
        'Source': 'https://github.com/materialslab/grainstat',
        'Documentation': 'https://grainstat.readthedocs.io/',
    },
)