from setuptools import setup

setup(
    name="text_generation_app",
    version="1.0",
    description="A text generation application",
    author="Your Name",
    author_email="your_email@example.com",
    install_requires=[
        "torch",
        "transformers"
    ],
    entry_points={
        "console_scripts": [
            "text_generation_app=main:main"
        ]
    },
)
