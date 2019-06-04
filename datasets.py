import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms


class CaptionDataset(Dataset):
    def __init__(self, image_dir, json_annotation_file, split, transform=None):
        # todo: implement different training/val/test splits
        if transform is not None:
            image_transform = transforms.Compose([
                transform,
                transforms.ToTensor()
            ])
        else:
            image_transform = transforms.ToTensor()
        self.captions = dset.CocoCaptions(root=image_dir,
                                annFile=json_annotation_file,
                                transform=image_transform)

    def __len__(self):
        return len(self.captions)

    def __getitem(self, index):
        # todo: need to return img, caption, caption_length, all_captions_for_img when in eval mode
        img, caption = self.captions[index]
        return img, caption, len(caption)


# class CaptionDataset(Dataset):
#     """
#     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
#     """
#
#     def __init__(self, data_folder, data_name, split, transform=None):
#         """
#         :param data_folder: folder where data files are stored
#         :param data_name: base name of processed datasets
#         :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
#         :param transform: image transform pipeline
#         """
#         self.split = split
#         assert self.split in {'TRAIN', 'VAL', 'TEST'}
#
#         # Open hdf5 file where images are stored
#         self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
#         self.imgs = self.h['images']
#
#         # Captions per image
#         self.cpi = self.h.attrs['captions_per_image']
#
#         # Load encoded captions (completely into memory)
#         with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
#             self.captions = json.load(j)
#
#         # Load caption lengths (completely into memory)
#         with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
#             self.caplens = json.load(j)
#
#         # PyTorch transformation pipeline for the image (normalizing, etc.)
#         self.transform = transform
#
#         # Total number of datapoints
#         self.dataset_size = len(self.captions)
#
#     def __getitem__(self, i):
#         # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
#         img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
#         if self.transform is not None:
#             img = self.transform(img)
#
#         caption = torch.LongTensor(self.captions[i])
#
#         caplen = torch.LongTensor([self.caplens[i]])
#
#         if self.split is 'TRAIN':
#             return img, caption, caplen
#         else:
#             # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
#             all_captions = torch.LongTensor(
#                 self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
#             return img, caption, caplen, all_captions
#
#     def __len__(self):
#         return self.dataset_size
