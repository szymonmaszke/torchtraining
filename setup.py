import functools
import os

import setuptools

exec(open("torchtrain/_version.py").read())  # get __version__
exec(open("torchtrain/_name.py").read())  # get _name

extras = {
    "horovod": ["horovod[pytorch]"],
    # Loggers
    "tensorboard": [
        "tensorboard>=2.3.0",
        "matplotlib>=3.3.1",  # add_figure
        "pillow>=7.2.0",  # add_image, add_images
        "moviepy>=1.0.3",  # add_video
    ],
    "neptune": ["neptune-client>=0.4.119"],
    "comet": ["comet-ml>=3.1.17"],
}


extras["callbacks"] = extras["tensorboard"] + extras["neptune"] + extras["comet"]
extras["accelerators"] = extras["horovod"]

extras["all"] = list(
    functools.reduce(
        lambda requirements1, requirements2: requirements1 + requirements2,
        extras.values(),
    )
)

extras["test"] = extras["callbacks"]

setuptools.setup(
    name=_name,
    version=__version__,
    license="MIT",
    author="Szymon Maszke",
    author_email="szymon.maszke@protonmail.com",
    description="Functional & flexible neural network training with PyTorch.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/torchtrain",
    packages=setuptools.find_packages(),
    install_requires=["torch>=1.3.0", "loguru>=0.5.1", "rich>=2.3.0", "PyYAML>=5.3.1"],
    extras_require=extras,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Website": "https://szymonmaszke.github.io/torchtrain",
        "Documentation": "https://szymonmaszke.github.io/torchtrain/#torchtrain",
        "Issues": "https://github.com/szymonmaszke/torchtrain/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc",
    },
    keywords="pytorch train functional flexible research fit epoch step simple fast",
)
