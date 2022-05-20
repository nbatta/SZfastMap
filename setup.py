from setuptools import setup
pname='SZFastMap'
setup(name=pname,
      version='0.1',
      description='Random SZ maps',
      #url='http://github.com/marcelo-alvarez/halosky',
      author='Marcelo Alvarez & Sigurd Naess & Nick Battaglia',
      license='MIT',
      packages=['szfastmap'],
      #package_data={
      #  pname: ["data/*"]
      #},
      zip_safe=False)