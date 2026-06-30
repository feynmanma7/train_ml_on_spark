import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
        ]
        build_args = ["--config", "Release" if not self.debug else "Debug"]
        build_args += ["--", "-j4"]

        build_dir = os.path.join(self.build_temp, "cmake_build")
        os.makedirs(build_dir, exist_ok=True)

        subprocess.check_call(["cmake", os.path.abspath(".")] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)

setup(
    name="torch_lite",
    version="0.1.0",
    author="torch_lite",
    description="A PyTorch-like deep learning library with C++ backend",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    packages=["torch_lite"],
    package_dir={"torch_lite": "torch_lite"},
    ext_modules=[CMakeExtension("torch_lite._torch_lite_core")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=["numpy", "pybind11>=2.10"],
    zip_safe=False,
)
