from setuptools import setup, find_packages

install_requires = [
    "cpprb>=8.1.1",
    "setuptools>=41.0.0",
    "numpy>=1.16.0",
    "joblib",
    "scipy",
    "tensorflow-probability==0.8.0"
]

extras_require = {
    "tf": ["tensorflow>=2.0.0"],
    "tf_gpu": ["tensorflow-gpu>=2.0.0"],
    "examples": ["gym[atari]", "opencv-python"],
    "test": ["coveralls", "gym[atari]", "matplotlib", "opencv-python"]
}

setup(
    name="tf2rl",
    version="0.1.19",
    description="Deep Reinforcement Learning for TensorFlow2",
    url="https://github.com/keiohta/tf2rl",
    author="Kei Ohta",
    author_email="dev.ohtakei@gmail.com",
    license="MIT",
    packages=find_packages("."),
    install_requires=install_requires,
    extras_require=extras_require)
