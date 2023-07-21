import setuptools


with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    include_package_data=True,
    name='OpenAEM',
    version='0.0.1',
    description='Synthetic velocity field induced by attached eddies',
    author='Duosi Fan',
    author_email='duosifan@hotmail.com',
    packages=setuptools.find_packages(),
    install_requires=required,
    long_description='some markdown',
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
    ],
)