#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

# 현재 디렉토리 경로
here = os.path.abspath(os.path.dirname(__file__))

# README 파일 읽기
with open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mejurix-medicallegal-embedder",
    version="0.1.0",
    author="Mejurix",
    author_email="contact@mejurix.com",
    description="의료-법률 도메인 특화 NER 기반 임베딩 모델",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mejurix/mejurix-medicallegal-embedder",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.12.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "scikit-learn>=0.24.0",
    ],
    include_package_data=True,
    package_data={
        'mejurix_medicallegal_embedder': ['pretrained/model/**/*']
    },
    entry_points={
        'console_scripts': [
            'medicallegal-embedder=mejurix_medicallegal_embedder.cli:main',
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/mejurix/medicallegal-embedder/issues",
        "Documentation": "https://github.com/mejurix/medicallegal-embedder",
        "Source Code": "https://github.com/mejurix/medicallegal-embedder",
    },
    keywords="embedding, ner, medical, legal, nlp, transformer",
) 