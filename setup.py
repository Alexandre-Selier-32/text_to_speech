from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

print(find_packages())

setup(name='text-to-speech',
      version="0.0.12",
      description="Text To Speec Model",
      license="MIT",
      author="TTS-team",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
