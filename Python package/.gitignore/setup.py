from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup (name = 'styletx',
       version = '1.0.1.1',
       description = 'Initial release with only Style Transfer',
       packages = ['styletx'],
       author = 'Dinesh Kumar Gnanasekaran',
       author_email = 'dinesh.gna111@gmail.com',
       long_description = long_description,
       long_description_content_type = 'text/markdown',
       lecense = 'MIT',
       zip_safe = False,
       include_package_data=True)