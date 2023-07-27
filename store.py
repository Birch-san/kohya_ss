import copy
import io
import json
import random
import typing
from datetime import datetime
from typing import Dict, Generator, List, Tuple
import os

import gridfs
import torch
import tqdm
from PIL import Image
from pymongo import MongoClient
from scipy.interpolate import interp1d

###############
# setup: dotenv
###############
# fmt: off
mongo_connect_string = str(os.environ['MONGO_URI'])
aspect_f = {"buckets": [[704, 1408], [512, 960], [640, 1152], [704, 1216], [704, 1152], [448, 704], [768, 1152], [448, 640], [512, 704], [832, 1088], [768, 960], [640, 768], [832, 960], [576, 640], [832, 896], [1024, 1024], [896, 832], [640, 576], [960, 832], [768, 640], [960, 768], [1088, 832], [704, 512], [640, 448], [1152, 768], [704, 448], [1152, 704], [1216, 704], [1152, 640], [960, 512], [1408, 704]], "bucket_ratios": [0.5, 0.5333333333333333, 0.5555555555555556, 0.5789473684210527, 0.6111111111111112, 0.6363636363636364, 0.6666666666666666, 0.7, 0.7272727272727273, 0.7647058823529411, 0.8, 0.8333333333333334, 0.8666666666666667, 0.9, 0.9285714285714286, 1.0, 1.0769230769230769, 1.1111111111111112, 1.1538461538461537, 1.2, 1.25, 1.3076923076923077, 1.375, 1.4285714285714286, 1.5, 1.5714285714285714, 1.6363636363636365, 1.7272727272727273, 1.8, 1.875, 2.0]}
aspect = list(zip(aspect_f["bucket_ratios"], aspect_f["buckets"]))
client = MongoClient(mongo_connect_string, w=1)
# fmt: on
###############
# Tensor Silliness
###############
class StringArray:
    def __init__(self, strings : typing.List[str], encoding : typing.Literal['ascii', 'utf_16_le', 'utf_32_le'] = 'utf_16_le'):
        strings = list(strings)
        self.encoding = encoding
        self.multiplier = dict(ascii = 1, utf_16_le = 2, utf_32_le = 4)[encoding]
        self.data = torch.ByteTensor(torch.ByteStorage.from_buffer(''.join(strings).encode(encoding)))
        self.cumlen = torch.LongTensor(list(map(len, strings))).cumsum(dim = 0).mul_(self.multiplier)

    def __getitem__(self, i):
        return bytes(self.data[(self.cumlen[i - 1] if i >= 1 else 0) : self.cumlen[i]]).decode(self.encoding)

    def __len__(self):
        return len(self.cumlen)

    def tolist(self):
        data_bytes, cumlen = bytes(self.data), self.cumlen.tolist()
        return [data_bytes[0:cumlen[0]].decode(self.encoding)] + [data_bytes[start:end].decode(self.encoding) for start, end in zip(cumlen[:-1], cumlen[1:])]
###############
# Dataset Preparation
###############
def find_aspect(width: int, height: int) -> Tuple[int,int]:
    ratio = width / height
    ratio = min(aspect, key=lambda x:abs(x[0]-ratio))
    return (ratio[1][0], ratio[1][1])
def generate_images() -> Generator[Tuple[str, str, float], None, None]:
    db = client.database
    col = db.dataset
    ########################
    # danbooru2022
    ########################
    for i in col.find(
        {
            "source_name": "danbooru2022",
            "params.size": { "$exists": True },
        },
        { "source_name": 1, "source_id": 1, "params.size": 1 },
        sort=[("insert_date", -1)]
    ):
        size = i["params"]["size"]
        yield (i["source_name"], i["source_id"], find_aspect(size["width"], size["height"]), (size["width"], size["height"]))
    ########################
    # danbooru2021
    ########################
    for i in col.find(
        {
            "source_name": "danbooru2021",
            "params.size": { "$exists": True },
        },
        { "source_name": 1, "source_id": 1, "params.size": 1 },
        sort=[("insert_date", -1)]
    ):
        size = i["params"]["size"]
        yield (i["source_name"], i["source_id"], find_aspect(size["width"], size["height"]), (size["width"], size["height"]))
    #pixiv
    for i in col.find(
        {
            "source_name": "pixiv",
            "params.size": { "$exists": True },
        },
        { "source_name": 1, "source_id": 1, "params.size": 1},
        sort=[("insert_data", -1)]
    ):
        size = i["params"]["size"]
        yield (i["source_name"], i["source_id"], find_aspect(size["width"], size["height"]), (size["width"], size["height"]))
    #synthetic
###############
# ImageStore
###############
class ImageStore:
    def __init__(self) -> None:
        self.image_files = []
        self.image_files.extend(generate_images())

    def __len__(self) -> int:
        return len(self.image_files)

    # iterator returns height/width of images and their index in the store
    def entries_iterator(self) -> Generator[Tuple[Tuple[int, int], int], None, None]:
        for f in range(len(self)):
            yield self.image_files[f][3], f

class AspectBucket:
    def __init__(self, store: ImageStore,
                 batch_size: int,
                 max_ratio: float = 2):

        self.batch_size = batch_size
        self.total_dropped = 0

        if max_ratio <= 0:
            self.max_ratio = float('inf')
        else:
            self.max_ratio = max_ratio

        self.store = store
        self.buckets = []
        self._bucket_ratios = []
        self._bucket_interp = None
        self.bucket_data: Dict[tuple, List[int]] = dict()
        self.init_buckets()
        self.fill_buckets()

    def init_buckets(self):
        self.buckets = aspect_f["buckets"]

        # cache the bucket ratios and the interpolator that will be used for calculating the best bucket later
        # the interpolator makes a 1d piecewise interpolation where the input (x-axis) is the bucket ratio,
        # and the output is the bucket index in the self.buckets array
        # to find the best fit we can just round that number to get the index
        self._bucket_ratios = aspect_f["bucket_ratios"]
        self._bucket_interp = interp1d(self._bucket_ratios, list(range(len(self.buckets))), assume_sorted=True,
                                       fill_value=None)

        # convert buckets from lists (from the json) to tuples
        self.buckets = list(map(lambda x: tuple(x), self.buckets))

        for b in self.buckets:
            self.bucket_data[b] = []

    def get_batch_count(self):
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())

    def get_bucket_info(self):
        return json.dumps({ "buckets": self.buckets, "bucket_ratios": self._bucket_ratios })

    def get_batch_iterator(self) -> Generator[Tuple[Tuple[int, int, int]], None, None]:
        """
        Generator that provides batches where the images in a batch fall on the same bucket

        Each element generated will be:
            (index, w, h)

        where each image is an index into the dataset
        :return:
        """
        max_bucket_len = max(len(b) for b in self.bucket_data.values())
        index_schedule = list(range(max_bucket_len))
        random.shuffle(index_schedule)

        bucket_len_table = {
            b: len(self.bucket_data[b]) for b in self.buckets
        }

        bucket_schedule = []
        for i, b in enumerate(self.buckets):
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))

        random.shuffle(bucket_schedule)

        bucket_pos = {
            b: 0 for b in self.buckets
        }

        total_generated_by_bucket = {
            b: 0 for b in self.buckets
        }

        for bucket_index in bucket_schedule:
            b = self.buckets[bucket_index]
            i = bucket_pos[b]
            bucket_len = bucket_len_table[b]

            batch = []
            while len(batch) != self.batch_size:
                # advance in the schedule until we find an index that is contained in the bucket
                k = index_schedule[i]
                if k < bucket_len:
                    entry = self.bucket_data[b][k]
                    batch.append(entry)

                i += 1

            total_generated_by_bucket[b] += self.batch_size
            bucket_pos[b] = i
            yield [(idx, *b) for idx in batch]

    def fill_buckets(self):
        entries = self.store.entries_iterator()
        total_dropped = 0

        for entry, index in tqdm.tqdm(entries, total=len(self.store)):
            if not self._process_entry(entry, index):
                total_dropped += 1

        for b, values in self.bucket_data.items():
            # shuffle the entries for extra randomness and to make sure dropped elements are also random
            random.shuffle(values)

            # make sure the buckets have an exact number of elements for the batch
            to_drop = len(values) % self.batch_size
            self.bucket_data[b] = list(values[:len(values) - to_drop])
            total_dropped += to_drop

        self.total_dropped = total_dropped

    def _process_entry(self, entry: Tuple[int, int], index: int) -> bool:
        aspect = entry[0] / entry[1] # width / height

        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            return False

        best_bucket = self._bucket_interp(aspect)

        if best_bucket is None:
            return False

        bucket = self.buckets[round(float(best_bucket))]

        self.bucket_data[bucket].append(index)

        return True


Image.MAX_IMAGE_PIXELS = None
###############
# AspectDataset
###############
class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, bucket: AspectBucket, ucg: float = 0.1):
        self.ucg = ucg
        self.data_source_name = StringArray(map(lambda x:x[0], bucket.store.image_files))
        self.data_source_id = StringArray(map(lambda x:x[1], bucket.store.image_files))
        self.len = len(bucket.store)
        self.client = MongoClient(mongo_connect_string, w=1)
        self.db = self.client.database
        self.fs = gridfs.GridFS(self.db, collection="dataset_files")

    def __len__(self):
        return self.len

    # 0 = resize
    # 1 = crop and resize
    # 2 = resize and fill
    def resize_image(self, im, width, height):
        LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return res, (width // 2 - src_w // 2, height // 2 - src_h // 2)

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {'latent': None, 'captions': None, "original_sizes_hw": None}
        #image = self.data[item[0]]
        captions = []
        source_name = self.data_source_name[item[0]]
        source_id = self.data_source_id[item[0]]

        f = self.fs.find_one({
            "metadata.source_name": source_name,
            "metadata.source_id": source_id
        })

        return_dict['latent'] = f.read() if f else None

        col = self.db.dataset
        p = col.find_one({ "source_name": source_name, "source_id": source_id })

        # tags
        if 'tags' in p["captions"]:
            captions.extend(map(lambda x:x["name"].replace('_', ' '), p["captions"]["tags"]))
        # rating
        if 'rating' in p["captions"]:
            r = f'{p["captions"]["rating"]["name"]}'
            captions.append(r)
            if r == 'questionable' or r == 'explicit':
                captions.append('nsfw')

        # date gradient
        date = p["creation_date"]
        if date >= datetime(1995, 1, 1) and date < datetime(2010, 1, 1):
            captions.append('oldest')
        elif date >= datetime(2010, 1, 1) and date < datetime(2015, 1, 1):
            captions.append('old')
        elif date >= datetime(2015, 1, 1) and date < datetime(2020, 1, 1):
            captions.append('new')
        elif date >= datetime(2020, 1, 1):
            captions.append('newest')

        ########################
        # danbooru2022
        ########################
        if source_name == 'danbooru2022':
            captions.extend(map(lambda x:x.replace('_', ' '), p["metadata"]["tags"]))
            r = p["metadata"]["content-rating"].lower()
            captions.append(r)
            if r == 'questionable' or r == 'explicit':
                captions.append('nsfw')
            captions.append('danbooru')
        ########################
        # danbooru2021
        ########################
        if source_name == 'danbooru2021':
            # original tags
            captions.extend(map(lambda x:x.replace('_', ' '), p["metadata"]["tag_string"].split()))
            # quality gradient
            score = int(p["metadata"]["score"])
            if score >= 150:
                captions.append('masterpiece')
            elif score >= 100 and score < 150:
                captions.append('best quality')
            elif score >= 75 and score < 100:
                captions.append('high quality')
            elif score >= 25 and score < 75:
                captions.append('medium quality')
            elif score >= 0 and score < 25:
                captions.append('normal quality')
            elif score < 0 and score >= -5:
                captions.append('low quality')
            elif score < -5:
                captions.append('worst quality')
            # rating gradient
            if p["metadata"]['rating'] == 'q':
                captions.append('questionable')
                captions.append('nsfw')
            elif p["metadata"]['rating'] == 'e':
                captions.append('explicit')
                captions.append('nsfw')
            elif p["metadata"]['rating'] == 's':
                captions.append('sensitive')
            else:
                captions.append('general')
            # deleted metric
            if p["metadata"]["is_deleted"]:
                captions.append("deleted")
        if source_name == "pixiv":
            captions.extend(
                map(
                    lambda x:x["name"].replace('_', ' '),
                    filter(lambda x:x["name"] is not None, p["metadata"]["tags"])
                )
            )
            captions.extend(
                map(
                    lambda x:x["translated_name"].replace('_', ' '),
                    filter(lambda x:x["translated_name"] is not None, p["metadata"]["tags"])
                )
            )
            captions.append(p["metadata"]["user"]["name"])
            captions.append(p["metadata"]["user"]["account"])

        caption_file = list(set(captions))
        random.shuffle(caption_file)

        return_dict['source_name'] = source_name
        return_dict['source_id'] = source_id

        return_dict["original_sizes_hw"] = None
        return_dict["crop_top_lefts"] = None
        return_dict["target_sizes_hw"] = None

        return_dict['captions'] = caption_file

        if return_dict["latent"] is not None:
            crop_to_w = item[1]
            crop_to_h = item[2]
            im = Image.open(io.BytesIO(return_dict["latent"])).convert('RGB')
            return_dict["original_sizes_hw"] = [im.height, im.width]
            return_dict["target_sizes_hw"] = [crop_to_h, crop_to_w]
            byte_arr = io.BytesIO()
            im, crop_pos = self.resize_image(im, crop_to_w, crop_to_h)
            im.save(byte_arr, format="PNG")
            return_dict["crop_top_lefts"] = [crop_pos[1], crop_pos[0]]
            return_dict["latent"] = byte_arr.getvalue()
        else:
            return_dict["captions"] = None

        return (
            copy.deepcopy(return_dict["latent"]),
            copy.deepcopy(return_dict["captions"]),
            copy.deepcopy(return_dict["source_name"]),
            copy.deepcopy(return_dict["source_id"]),
            copy.deepcopy(return_dict["original_sizes_hw"]),
            copy.deepcopy(return_dict["target_sizes_hw"]),
            copy.deepcopy(return_dict["crop_top_lefts"])
        )
###############
# AspectBucketSampler
###############
class AspectBucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket: AspectBucket, dataset: AspectDataset):
        super().__init__(None)
        self.bucket = bucket
        self.sampler = dataset

    def __iter__(self):
        # return batches as they are and let accelerate distribute
        indices = self.bucket.get_batch_iterator()
        return iter(indices)

    def __len__(self):
        return self.bucket.get_batch_count()

if __name__ == '__main__':
    bs=1
    bucket = AspectBucket(ImageStore(), bs)
    dataset = AspectDataset(bucket)
    sampler = AspectBucketSampler(bucket=bucket, dataset=dataset)
    from torch.utils.data import DataLoader
    train_dl = DataLoader(dataset, batch_size=bs, sampler=sampler)
    for x in train_dl:
        print(x[1])
        break
