from setuptools import setup, find_packages

setup(
    name="evalforge",
    version="0.1.0",
    description="Open Source Model Health & Evaluation Intelligence Engine",
    author="Varun Rao",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
    ],
    python_requires=">=3.8",
)