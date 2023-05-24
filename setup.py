from setuptools import setup

setup(name='model-zoo-jax',
      version='1.0.0',
      packages=["model_zoo_jax"],
      install_requires=[
        "chex",
        "datasets",
        "dm-haiku",
        "dm-pix",
        "jax",
        "jaxlib",
        "matplotlib",
        "numpy",
        "optax",
        "pandas",
        "wandb",
        ],
    )
