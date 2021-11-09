from distutils.core import setup

setup(name='BLCC',
      version='1.0',
      description='Bayesian Latent Class Clustering',
      url='https://github.com/smturro2/URES-Project-2',
      author='Scott Turro',
      author_email='turroscott@gmail.com',
      packages=['BLCC'],
      package_dir={'': 'src_python'},
      install_requires=['pandas',"numpy","scipy"]
     )