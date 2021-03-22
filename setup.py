
from setuptools import setup, find_packages

setup(
    name='irony',
    version='1.0.0',
    url='https://github.com/credwood/irony.git',
    author='Charysse Redwood',
    author_email='charysse.redwood@gmail.com',
    description='is this a joke? query GPT-2 to find out',
    packages=find_packages(),
    install_requires=[
        'aiofiles==0.6.0',
        'fastapi==0.63.0',
        'Jinja2==2.11.3',
        'numpy==1.18.5',
        'pydantic==1.8.1',
        'python-multipart==0.0.5',
        'scipy==1.4.1',
        'tokenizers==0.9.4',
        'torch==1.7.0',
        'torchvision==0.8.1',
        'transformers==4.0.0',
        'typing==3.7.4.3',
        'uvicorn==0.13.4',
    ],
)