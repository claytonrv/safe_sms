
from codecs import open
from os import path
from setuptools import find_packages, setup

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='safe_sms',
    packages=find_packages(include=['safe_sms']),
    version='0.0.1',
    description='SMS spam/ham classifier lib',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='claytonrv',
    license='MIT',
    install_requires=[]
)