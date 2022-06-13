from setuptools import setup, find_packages

setup(
  name = 'deepvit-flax',
  packages = find_packages(exclude=['examples']),
  version = 'v0.0.3',
  license='MIT',
  description = 'Deep Vision Transformer (DeepViT) - Flax',
  author = 'Enrico Shippole',
  author_email = 'enricoship@gmail.com',
  url = 'https://github.com/conceptofmind/DeepViT-flax',
  keywords = [
    'artificial intelligence',
    'machine learning',
    'deep learning',
    'attention mechanism',
    'image recognition'
  ],
  install_requires=[
    'jax>=0.3.4',
    'einops>=0.4.1',
    'flax>=0.5.0',
    'optax>=0.1.2',
    'numpy>=1.21.2',
    'datasets>=2.2.2',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)