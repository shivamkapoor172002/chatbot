from setuptools import setup, find_packages

setup(
    name='streamlit-chat-app',
    version='1.0.0',
    author='Your Name',
    author_email='your@email.com',
    description='Streamlit Chat App',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'transformers',
        'torch',
        'sentence-transformers',
        # Add other dependencies here
    ],
)
