from distutils.core import setup,Extension

module1 = Extension('ext_module',
    include_dirs=['usr/local/include'],
    libraries=['pthread'],
    sources=['ext_module.c']
)

setup (name='ext_module',
    version='1.0',
    description='test python - C',
    author='DevpCondor',
    ext_modules=[module1]
)