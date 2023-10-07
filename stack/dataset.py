import h5py
import torch
import numpy as np
from brainbox.datasets.transforms import ClipRandomHorizontalFlip


class NaturalDataset(torch.utils.data.Dataset):

    def __init__(self, root="/home/datasets/natural", train=True, dt=30, flip=True, filtered=True):
        self._root = root
        self._dt = dt
        self._flip = flip
        self._filtered = filtered

        self._flipper = ClipRandomHorizontalFlip() if flip else None

        # Load dataset
        self._dataset = self._load_dataset(train=train)
        self._pre_process()
        self._dataset = self._dataset[:, :, ::4]

        # Load idxs
        n_batch = self._dataset.shape[0]
        n_steps = int(self._dataset.shape[2] / dt)
        self._idxs = [(b, t) for b in range(n_batch) for t in range(n_steps)]

    def __getitem__(self, i):
        b, i = self._idxs[i]
        clip = self._dataset[b, :, i*self._dt: (i+1)*self._dt]

        if self._flip:
            clip = self._flipper(clip)

        return clip, torch.rand(1)

    def __len__(self):
        return len(self._idxs)

    @property
    def hyperparams(self):
        return {}

    def _pre_process(self):
        self._dataset = self._dataset[:, :, :, 2:-2, 8:-8]
        if self._filtered:
            self._dataset.clamp_(min=-4000, max=4000)
            self._dataset.sub_(-0.2285)
            self._dataset.divide_(634.1161)
        else:
            self._dataset.sub_(115.9144)
            self._dataset.divide_(63.3384)

    def _load_dataset(self, train):
        if self._filtered:
            hf = h5py.File(f"{self._root}/filtered_natural.hdf5", "r")
        else:
            hf = h5py.File(f"{self._root}/natural.hdf5", "r")
        dataset_name = "train" if train else "test"
        dataset = np.array(hf.get(dataset_name))
        hf.close()

        dataset = torch.from_numpy(dataset)
        dataset = dataset.unsqueeze(1)
        dataset = dataset.type(torch.FloatTensor)

        print(f"{dataset.shape}")

        return dataset
