from setuptools import setup

setup(
    name="colab-chat",
    version="0.1.0",
    py_modules=["chat"],
    install_requires=[
        "transformers",
        "torch",
        "accelerate",
    ],
    entry_points={
        "console_scripts": [
            "colab_chat=chat:main",
        ],
    },
    author="Ashraff Hathibelagal",
    description="A minimal CLI chat application using transformers with core tool-use support.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
