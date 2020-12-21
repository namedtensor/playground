from setuptools import setup

setup(
    name="playground",
    version="0.1",
    packages=["playground"],
    package_data={"playground": []},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
