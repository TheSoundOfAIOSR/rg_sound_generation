import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mapping_models",
    version="0.0.1",
    author="Sound Generation OSR",
    author_email="amit.yadav.iitr@gmail.com",
    description="mapping models module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheSoundOfAIOSR/rg_sound_generation/mapping_models",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "ddsp==1.0.1",
        "click==7.1.2"
    ]
)
