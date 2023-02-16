from setuptools import setup, find_packages
with open('README.md', 'r') as fh:
    long_description = fh.read()
setup(
    name='dicee',
    description='Dice embedding is an hardware-agnostic framework for large-scale knowledge graph embedding applications',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['pandas>=1.5.1',
                      'modin[ray]>=0.16.2',
                      'polars>=0.15.13',
                      'pyarrow>=8.0.0',
                      'torch>=1.13.0',
                      'pytorch-lightning>=1.6.4',
                      'scikit-learn>=1.1.1',
                      'pytest>=6.2.5',
                      'gradio>=3.0.17'
                      ],
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    url='https://github.com/dice-group/dice-embeddings',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License"],
    python_requires='>==3.9.12',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
