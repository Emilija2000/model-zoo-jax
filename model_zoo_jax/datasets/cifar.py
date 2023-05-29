# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stolen shamelessly and only slightly modified, from this link:
#   https://github.com/google/jax/blob/main/examples/datasets.py
"""Datasets used in examples."""


import pickle
import os
from os import path
import urllib.request

import jax.numpy as np
from jax.random import permutation
import tarfile


_DATA = "./data/"


def download_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = _DATA +"cifar-10-python.tar.gz"

    if not os.path.exists(filename):
        print("Downloading CIFAR-10 dataset...")
        try:
            urllib.request.urlretrieve(url, filename)
        except urllib.error.URLError as e:
            print("Failed to download the CIFAR-10 dataset:", e)
            return False
        print("Download complete.")
    else:
        print("CIFAR-10 dataset is already downloaded.")

    return True

def extract_tar(filename, folder):
    if not os.path.exists(folder):
        print("Extracting CIFAR-10 dataset...")
        try:
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall()
        except tarfile.ReadError as e:
            print("Failed to extract the CIFAR-10 dataset:", e)
            return False
        print("Extraction complete.")
    else:
        print("CIFAR-10 dataset is already extracted.")

    return True

def unpickle(file):
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def load_cifar10():
    folder = _DATA+"cifar-10-batches-py"
    print('here')
    # Load training data
    train_images, train_labels = [], []
    for batch in range(1, 6):
        batch_file = os.path.join(folder,f"data_batch_{batch}")
        if not os.path.exists(batch_file):
            print(f"Missing {batch_file}. Make sure you have downloaded the CIFAR-10 dataset.")
            return None

        batch_data = unpickle(batch_file)
        train_images.append(batch_data[b"data"])
        train_labels.extend(batch_data[b"labels"])

    train_images = np.concatenate(train_images)
    train_images = train_images.reshape(-1, 3, 32, 32)
    train_images = np.transpose(train_images, (0, 2, 3, 1))
    train_labels = np.array(train_labels)

    # Load test data
    test_file = os.path.join(folder, "test_batch")
    if not os.path.exists(test_file):
        print(f"Missing {test_file}. Make sure you have downloaded the CIFAR-10 dataset.")
        return None

    test_data = unpickle(test_file)
    test_images = test_data[b"data"]
    test_labels = np.array(test_data[b"labels"])
    test_images = test_images.reshape(-1, 3, 32, 32)
    test_images = np.transpose(test_images, (0, 2, 3, 1))

    return train_images, train_labels, test_images, test_labels


def cifar10_raw():
    """Download and parse the raw CIFAR10 dataset."""
    tar_filename = _DATA+"cifar-10-python.tar.gz"
    extract_folder = _DATA

    if download_cifar10():
        if extract_tar(tar_filename, extract_folder):
            cifar10_data = load_cifar10()
            if cifar10_data is not None:
                train_images, train_labels, test_images, test_labels = cifar10_data
                print("Train images shape:", train_images.shape)
                print("Train labels shape:", train_labels.shape)
                print("Test images shape:", test_images.shape)
                print("Test labels shape:", test_labels.shape)
                return train_images, train_labels, test_images, test_labels
