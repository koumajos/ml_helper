import setuptools

setuptools.setup(
    name="ml_helper",
    version="0.01",
    author="Josef Koumar",
    author_email="koumajos@fit.cvut.cz",
    description="Helper with naive usage of ML models.",
    packages=[
        "ml_helper",
    ],
    install_requires=[
        "numpy>=1.22.2",
        "scipy>=1.6.3",
        "matplotlib>=3.4.3",
    ],
    package_dir={"": "src"},
    python_requires=">=3.10",
)
