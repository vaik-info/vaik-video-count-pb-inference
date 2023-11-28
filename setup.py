import os
import setuptools


def load_requires_from_file(fname):
    if not os.path.exists(fname):
        print(f'Pass: {fname}')
        return []
    return [pkg.strip() for pkg in open(fname, 'r')]


def load_links_from_file(filepath):
    res = []
    with open(filepath) as fp:
        for pkg_name in fp.readlines():
            if "git+ssh" in pkg_name:
                res.append(pkg_name[pkg_name.find("git+ssh"):].strip())
    return res


if __name__ == '__main__':
    setuptools.setup(
        name="vaik-video-count-pb-inference",
        version="1.0.0",
        url="https://github.com/vaik-info/vaik-video-count-pb-inference.git",
        install_requires=load_requires_from_file('requirements.txt'),
        dependency_links=load_links_from_file('requirements.txt'),
        author="vaik-info",
        author_email="info@vaikobo.com",
        description="Inference with the count for video PB model.",
        long_description="Inference with the count for video PB model.",
        long_description_content_type="text/markdown",
        license='MIT',
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3.8.10",
            'License :: OSI Approved :: MIT License',
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
    )