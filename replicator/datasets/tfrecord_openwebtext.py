import tensorflow as tf
# for making tf.data.Dataset to return numpy arrays
import tensorflow_datasets as tfds

from transformers import GPT2TokenizerFast

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import IterableDataset

import pytorch_lightning as pl


train_filenames = [
    "gs://fengwenfeng-nlp/openwebtext/openwebtext_0_0_10000.tfrecords",
    "gs://fengwenfeng-nlp/openwebtext/openwebtext_1_0_10000.tfrecords",
    "gs://fengwenfeng-nlp/openwebtext/openwebtext_2_0_10000.tfrecords",
    "gs://fengwenfeng-nlp/openwebtext/openwebtext_3_0_10000.tfrecords",
]

val_filenames = [
    "gs://fengwenfeng-nlp/openwebtext/openwebtext_4_0_10000.tfrecords",
]


def tfrecord_parse_func(example_proto):
    features = {
        "text": tf.io.VarLenFeature(tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])), tf.int32)


def get_dataset(filenames, parse_func, batch_size=16, repeat=False, cache=False):
    ds = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=tf.data.AUTOTUNE)
    if cache:
        df = df.cache()
    if repeat:
        ds = ds.repeat()
    ds = ds.map(parse_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = tfds.as_numpy(ds)
    return ds


class TFRecordIterableDataset(IterableDataset):
    def __init__(self, filenames, parse_func, batch_size=16, repeat=False, cache=False, items_per_file=10000):
        self.ds = get_dataset(filenames, parse_func, batch_size, repeat, cache)
        self.num_items = len(filenames) * items_per_file
        self.batch_size = batch_size
        self._iterator = None

        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '<pad>', 'unk_token': '<unk>', 'mask_token': '<mask>'})
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()


    def __iter__(self):
        self._iterator = iter(self.ds)
        return self._iterator

    # def __next__(self):
    #   return next(self._iterator)

    def __len__(self):
        num_batches = self.num_items // self.batch_size
        if self.num_items % self.batch_size == 0:
            return num_batches
        else:
            return num_batches + 1

    def vocab_size(self):
        return len(self.vocab)


class TFRecordDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, seq_len: int, train_filenames, val_filenames, tfrecord_parse_func):
        super().__init__()
        self.batch_size, self.seq_len = batch_size, seq_len
        self.train_filenames, self.val_filenames, self.tfrecord_parse_func = train_filenames, val_filenames, tfrecord_parse_func

    # def setup(self):
        self.train_data = TFRecordIterableDataset(
            self.train_filenames, self.tfrecord_parse_func, batch_size=self.batch_size)
        self.val_data = TFRecordIterableDataset(
            self.train_filenames, self.tfrecord_parse_func, batch_size=self.batch_size)
        # self.vocab_size = self.train_data.vocab_size()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=None,
            # collate_fn=self.mask
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None,
            # collate_fn=self.mask
        )
