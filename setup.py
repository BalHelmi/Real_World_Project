from setuptools import find_packages,setup

from typing import List

hypha_e = '-e .'

def get_requirements(file:str)->List[str]:
    requirements = []
    with open(file) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if hypha_e in requirements :
            requirements.remove(hypha_e)
    
    return requirements

setup(
    name = "Real World Project",
    version = "0.0.1",
    author = "Helmi Balhoudi",
    author_email = "balhoudihelmi@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)