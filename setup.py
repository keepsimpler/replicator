import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="replicator",
    version="0.0.1",
    author="fengwenfeng",
    author_email="fengwenfeng@gmail.com",
    description="Replicator dynamics view of deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keepsimpler/replicator",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "torch >=1.8",
        "einops >=0.3",
        "pytorch_lightning",
        "transformers",
    ]
)