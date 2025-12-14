from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="transmargan",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="TransMAR-GAN: Transformer-based Multi-Scale Adversarial Reconstruction GAN for Metal Artifact Reduction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TransMAR-GAN",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/TransMAR-GAN/issues",
        "Documentation": "https://github.com/yourusername/TransMAR-GAN/docs",
        "Source Code": "https://github.com/yourusername/TransMAR-GAN",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(exclude=["tests", "docs", "scripts", "results"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "transmargan-train=training.train_combined:main",
            "transmargan-test=testing.test_finetuned_model:main",
        ],
    },
)
