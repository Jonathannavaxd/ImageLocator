from setuptools import setup, find_packages

setup(
    name='pyimagelocator',
    version='0.1.0',
    author='Jhonatan Navarro',
    author_email='Jonathannavaxd@gmail.com',
    description='It was created out of the need to locate RGBA images with alpha background in given images.',
    packages=find_packages(),
    install_requires=[],  # Aquí puedes agregar dependencias si las tienes
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Windows',
    ],
    python_requires='>=3.12.4',
)