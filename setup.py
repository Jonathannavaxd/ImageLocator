from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyimagelocator',
    version='0.1.0',
    author='Jhonatan Navarro',
    author_email='Jonathannavaxd@gmail.com',
    description='It was created out of the need to locate RGBA images with alpha background in given images.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Jonathannavaxd/ImageLocator',  # Cambia esto a la URL de tu proyecto
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Windows',
    ],
    python_requires='>=3.12.4',
)