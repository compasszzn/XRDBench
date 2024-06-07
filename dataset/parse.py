import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import sys

def load_dataset(name='test.tfrecord',):
    # Load the tfrecord file
    dataset = tf.data.TFRecordDataset([name])

    # Parse the dataset
    parsed_dataset = dataset.map(parse_example)

    # Create an instance of the dataset
    my_dataset = MyDataset(parsed_dataset)
    return my_dataset


# Create a dataset class
class MyDataset(Dataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = tf_dataset

    def __len__(self):
        return sum(1 for _ in self.tf_dataset)

    def __getitem__(self, index):
        for i, data in enumerate(self.tf_dataset):
            if i == index:
                intensity_list, spg_list, crysystem_list,element_list = data
                return {
                        # 'latt_dis': tensor_latt_dis,
                        'intensity': torch.tensor(intensity_list, dtype=torch.int64),
                        'spg': torch.tensor(spg_list, dtype=torch.int64),
                        'crysystem': torch.tensor(crysystem_list, dtype=torch.int64),
                        'element': torch.tensor(element_list, dtype=torch.int64)
                    }
                     
# Define a function to parse a single example
def parse_example(example_proto):
    feature_description = {
        'element_list': tf.io.FixedLenFeature([], tf.string),
        'latt_dis_list': tf.io.FixedLenFeature([], tf.string),
        'intensity_list': tf.io.FixedLenFeature([], tf.string),
        'spg_list': tf.io.FixedLenFeature([], tf.string),
        'crysystem_list': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    element_list = tf.io.parse_tensor(example['element_list'], out_type=tf.string)
    # latt_dis_list = tf.io.parse_tensor(example['latt_dis_list'], out_type=tf.float32)
    intensity_list = tf.io.parse_tensor(example['intensity_list'], out_type=tf.float32)
    spg_list = tf.io.parse_tensor(example['spg_list'], out_type=tf.int32)
    crysystem_list = tf.io.parse_tensor(example['crysystem_list'], out_type=tf.int32)
    return intensity_list, spg_list, crysystem_list,element_list


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()