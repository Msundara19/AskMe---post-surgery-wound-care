from setuptools import setup, find_packages
setup(
    name="meditrack",
    version="0.1.0",
    description="MediTrack: AI-powered wound healing monitoring",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
)
