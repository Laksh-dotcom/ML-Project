from setuptools import setup, find_packages


hyphen_e_dot = '-e.'
requirements = []

def get_requirements(filepath):
    with open(filepath, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

    if hyphen_e_dot in requirements:
        requirements.remove(hyphen_e_dot)
    
    return requirements

setup(
    name = "ML_Project",
    version = "0.0.1",
    author = "Lakshay",
    author_email = "sharlaksh584@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)