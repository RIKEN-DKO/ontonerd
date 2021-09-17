import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()
    
setuptools.setup(
        name="dkouqe",
        version="0.0.1",
        author="JC. Rangel",
        author_email="juliocesar.rangelreyes@riken.jp",
        description="My awesome package",
        long_description=long_description,
        long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        include=['dkoulinker', 'dataset_creation.*'])
    )
