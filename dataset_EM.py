from skimage import transform
import matplotlib.pyplot as plt
import os, glob, sys, mrc
from gaussian_filter import *
from numba import jit
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchBasedDataset:
    """
        ET, CT Data and other types of biomedical data are always much larger than natural images,
        so patch-based dataset should be used in this task.

        This part of code is borrowed from : Topaz, denoise3d.py
        https://github.com/tbepler/topaz , proposed by
        Tristan Bepler, Kotaro Kelley, Alex J. Noble, and Bonnie Berger, MIT
    """
    def __init__(self, tomo, patch_size=96, padding=32):
        self.tomo = tomo
        self.patch_size = patch_size
        self.padding = padding

        nz,ny,nx = tomo.shape

        pz = int(np.ceil(nz/patch_size))
        py = int(np.ceil(ny/patch_size))
        px = int(np.ceil(nx/patch_size))
        self.shape = (pz,py,px)
        self.num_patches = pz*py*px


    def __len__(self):
        return self.num_patches

    def __getitem__(self, patch):
        # patch index
        i, j, k = np.unravel_index(patch, self.shape)

        patch_size = self.patch_size
        padding = self.padding
        tomo = self.tomo

        # pixel index
        i = patch_size*i
        j = patch_size*j
        k = patch_size*k

        # make padded patch
        d = patch_size + 2*padding
        x = np.zeros((d, d, d), dtype=np.float32)

        # index in tomogram
        si = max(0, i-padding)
        ei = min(tomo.shape[0], i+patch_size+padding)
        sj = max(0, j-padding)
        ej = min(tomo.shape[1], j+patch_size+padding)
        sk = max(0, k-padding)
        ek = min(tomo.shape[2], k+patch_size+padding)

        # index in crop
        sic = padding - i + si
        eic = sic + (ei - si)
        sjc = padding - j + sj
        ejc = sjc + (ej - sj)
        skc = padding - k + sk
        ekc = skc + (ek - sk)

        x[sic:eic,sjc:ejc,skc:ekc] = tomo[si:ei,sj:ej,sk:ek]

        data = {'source': x, 'pos': np.array((i,j,k), dtype=int)}
        return data
        # return np.array((i,j,k), dtype=int),x

class Dataset3D(torch.utils.data.Dataset):
    """
        ET, CT Data and other types of biomedical data are always much larger than natural images,
        so patch-based dataset should be used in this task.

        This part of code is borrowed from : Topaz, denoise3d.py
        https://github.com/tbepler/topaz , proposed by
        Tristan Bepler, Kotaro Kelley, Alex J. Noble, and Bonnie Berger, MIT

    """
    def __init__(self, noisy_path, prior_path, tilesize = 96, swap_size = 3, swap_window_size = 5, swap_mode = 'internal',
                 swap_region = 'volume', swap_ratio = 0.1, N_train = 1000, N_test = 200, mode = 'train'):

        self.tilesize = tilesize
        self.N_train = N_train
        self.N_test = N_test
        self.swap_size = swap_size
        self.swap_window_size = swap_window_size
        self.swap_mode = swap_mode
        self.swap_ratio = swap_ratio
        self.swap_region = swap_region
        self.mode = mode

        self.noisy_paths = []
        self.prior_paths = []

        if swap_mode != 'cross' and swap_mode != 'internal':
            print('# Error: illegal swap mode, only cross mode or internal mode is accepted:', file=sys.stderr)
            print('# Forced to cross mode, Skipping...', file=sys.stderr)
            self.swap_mode = 'cross'

        if os.path.isfile(noisy_path) and os.path.isfile(prior_path):
            self.noisy_paths.append(noisy_path)
            self.prior_paths.append(prior_path)
        elif os.path.isdir(noisy_path) and os.path.isdir(prior_path):
            for epath in glob.glob(noisy_path + os.sep + '*'):
                name = os.path.basename(epath)
                opath = prior_path + os.sep + name
                if not os.path.isfile(opath):
                    print('# Error: name mismatch between noisy and prior directory,', name, file=sys.stderr)
                    print('# Skipping...', file=sys.stderr)
                else:
                    self.noisy_paths.append(epath)
                    self.prior_paths.append(opath)
        self.means = []
        self.stds = []
        self.noisy = []
        self.prior = []
        self.train_idxs = []
        self.test_idxs = []
        for i, (f_noisy, f_prior) in enumerate(zip(self.noisy_paths, self.prior_paths)):
            noisy = self.load_mrc(f_noisy)
            noisy = self.sampling(noisy)
            prior = self.load_mrc(f_prior)
            prior = self.sampling(prior)
            px = int(noisy.shape[0] / tilesize)
            py = int(noisy.shape[1] / tilesize)
            pz = int(noisy.shape[2] / tilesize)
            N_train = int(px*py*pz*0.9)
            N_test = int(px*py*pz*0.1)
            self.N_train = N_train
            self.N_test = N_test
            if noisy.shape != prior.shape:
                print('# Error: shape mismatch:', f_noisy, f_prior, file=sys.stderr)
                print('# Skipping...', file=sys.stderr)
            else:
                if noisy.shape[0] > noisy.shape[1]:
                    noisy = noisy.transpose(1, 0, 2)
                if prior.shape[0] > prior.shape[1]:
                    prior = prior.transpose(1, 0, 2)
                noisy_mean, noisy_std = self.calc_mean_std(noisy)
                prior_mean, prior_std = self.calc_mean_std(prior)
                self.means.append((noisy_mean, prior_mean))
                self.stds.append((noisy_std, prior_std))

                self.noisy.append(noisy)
                self.prior.append(prior)

                mask = np.ones(noisy.shape, dtype=np.uint8)
                train_idxs, test_idxs = self.sample_coordinates(mask, N_train, N_test,
                                                                vol_dims=(tilesize, tilesize, tilesize))

                self.train_idxs += train_idxs
                self.test_idxs += test_idxs

        # print(len(self.train_idxs))
        # print(len(self.test_idxs))

        if len(self.noisy) < 1:
            print('# Error: need at least 1 file to proceeed', file=sys.stderr)
            sys.exit(2)


    def sampling(self, tomo, scale=2, mode='nearest'):
        tomo = torch.from_numpy(tomo)
        tomo = tomo.unsqueeze(0).unsqueeze(0)
        if scale < 1:
            tomo = F.interpolate(tomo, size=(tomo.shape[2] * scale, tomo.shape[3], tomo.shape[4] * scale),
                                     mode=mode)
        else:
            tomo = F.interpolate(tomo, size=(int(tomo.shape[2] * scale), tomo.shape[3], int(tomo.shape[4] * scale)),
                                     mode=mode)
        tomo = tomo.squeeze(0).squeeze(0)
        tomo = np.array(tomo)
        return tomo

    def load_mrc(self, path):
        with open(path, 'rb') as f:
            content = f.read()
        tomo,_,_ = mrc.parse(content)
        tomo = tomo.astype(np.float32)
        return tomo

    def load_pydicom(self, path):
        return


    def sample_coordinates(self, mask, num_train_vols, num_val_vols, vol_dims=(96, 96, 96)):

        # This function is borrowed from:
        # https://github.com/juglab/cryoCARE_T2T/blob/master/example/generate_train_data.py
        """
        Sample random coordinates for train and validation volumes. The train and validation
        volumes will not overlap. The volumes are only sampled from foreground regions in the mask.

        Parameters
        ----------
        mask : array(int)
            Binary image indicating foreground/background regions. Volumes will only be sampled from
            foreground regions.
        num_train_vols : int
            Number of train-volume coordinates.
        num_val_vols : int
            Number of validation-volume coordinates.
        vol_dims : tuple(int, int, int)
            Dimensionality of the extracted volumes. Default: ``(96, 96, 96)``

        Returns
        -------
        list(tuple(slice, slice, slice))
            Training volume coordinates.
         list(tuple(slice, slice, slice))
            Validation volume coordinates.
        """

        dims = mask.shape
        cent = (np.array(vol_dims) / 2).astype(np.int32)
        mask[:cent[0]] = 0
        mask[-cent[0]:] = 0
        mask[:, :cent[1]] = 0
        mask[:, -cent[1]:] = 0
        mask[:, :, :cent[2]] = 0
        mask[:, :, -cent[2]:] = 0

        tv_span = np.round(np.array(vol_dims) / 2).astype(np.int32)
        span = np.round(np.array(mask.shape) * 0.1 / 2).astype(np.int32)
        val_sampling_mask = mask.copy()
        val_sampling_mask[:, :span[1]] = 0
        val_sampling_mask[:, -span[1]:] = 0
        val_sampling_mask[:, :, :span[2]] = 0
        val_sampling_mask[:, :, -span[2]:] = 0

        foreground_pos = np.where(val_sampling_mask == 1)
        sample_inds = np.random.choice(len(foreground_pos[0]), 2, replace=False)

        val_sampling_mask = np.zeros(mask.shape, dtype=np.int8)
        val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
        for z, y, x in zip(*val_sampling_inds):
            val_sampling_mask[z - span[0]:z + span[0],
            y - span[1]:y + span[1],
            x - span[2]:x + span[2]] = mask[z - span[0]:z + span[0],
                                       y - span[1]:y + span[1],
                                       x - span[2]:x + span[2]].copy()

            mask[max(0, z - span[0] - tv_span[0]):min(mask.shape[0], z + span[0] + tv_span[0]),
            max(0, y - span[1] - tv_span[1]):min(mask.shape[1], y + span[1] + tv_span[1]),
            max(0, x - span[2] - tv_span[2]):min(mask.shape[2], x + span[2] + tv_span[2])] = 0

        foreground_pos = np.where(val_sampling_mask)
        sample_inds = np.random.choice(len(foreground_pos[0]), num_val_vols,
                                       replace=num_val_vols < len(foreground_pos[0]))
        val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
        val_coords = []
        for z, y, x in zip(*val_sampling_inds):
            val_coords.append(tuple([slice(z - tv_span[0], z + tv_span[0]),
                                     slice(y - tv_span[1], y + tv_span[1]),
                                     slice(x - tv_span[2], x + tv_span[2])]))

        foreground_pos = np.where(mask)
        sample_inds = np.random.choice(len(foreground_pos[0]), num_train_vols,
                                       replace=num_train_vols < len(foreground_pos[0]))
        train_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
        train_coords = []
        for z, y, x in zip(*train_sampling_inds):
            train_coords.append(tuple([slice(z - tv_span[0], z + tv_span[0]),
                                       slice(y - tv_span[1], y + tv_span[1]),
                                       slice(x - tv_span[2], x + tv_span[2])]))

        return train_coords, val_coords

    def calc_mean_std(self, tomo):
        mu = tomo.mean()
        std = tomo.std()
        return mu, std

    def __len__(self):
        if self.mode == 'train':
            return self.N_train * len(self.noisy)
        else:
            return self.N_test * len(self.noisy)

    def __getitem__(self, idx):

        if self.mode == 'train':
            Idx = int(idx / self.N_train)
            idx = self.train_idxs[idx]
        else:
            Idx = int(idx / self.N_test)
            idx = self.test_idxs[idx]

        noisy = self.noisy[Idx]
        prior = self.prior[Idx]

        mean = self.means[Idx]
        std = self.stds[Idx]

        noisy_ = noisy[idx]
        prior_ = prior[idx]

        # noisy_ = (noisy_ - mean[0]) / std[0]
        noisy_ = (noisy_ - mean[0]) / std[0]
        prior_ = (prior_ - mean[1]) / std[1]
        noisy_, prior_ = self.augment(noisy_, prior_)

        noisy_ = np.expand_dims(noisy_, axis=0)
        prior_ = np.expand_dims(prior_, axis=0)

        swap_dims = (self.swap_size, self.swap_size, self.swap_size)
        swap_window_dims = (self.swap_window_size, self.swap_window_size, self.swap_window_size)
        swap_ratio = self.swap_ratio

        # gaussian_layer = nn.Sequential(GaussianDenoise3d(sigma=1))
        source, target, mask, center_mask_ = self.blind_spot_fusion(noisy_, prior_, swap_dims = swap_dims, swap_window_dims = swap_window_dims, swap_ratio = swap_ratio)

        source_ = torch.from_numpy(source).float()
        target_ = torch.from_numpy(target).float()
        mask_ = torch.from_numpy(mask).float()

        data = {'noisy':noisy_, 'source': source_, 'target': target_, 'mask': mask_, 'centers': center_mask_}

        return data

    @jit()
    def blind_spot_fusion(self, source, target, swap_dims = (3, 3, 3), swap_window_dims = (5, 5, 5), swap_ratio = 0.08):
        """
        Implementation for blind_spot pixel replacement in 3D image volume
        Args:
            source: noisy input volume reconstructed by SART mpi version with 19 threads or imod-WBP
            target: 2D denoised prior knowledge reconstructed by SART  or imod-WBP as source
            swap_dims: dimensionality for blind spot by 3D patch
            swap_window_dims: largest region for swapping
            swap_ratio: number of pixels (voxels) to be swapped

        Returns:
            output_source:
            output_target:
            swap_mask:
        """
        assert source.shape == target.shape     # keep the same dims

        swap_mode = self.swap_mode
        swap_region = self.swap_region

        dims = source.shape
        swap_mask = np.ones(dims)      # double mask for single blind-spot
        center_mask = np.zeros(dims)
        # gradient_mask = np.ones(dims)

        margin = (np.array(swap_window_dims) / 2).astype(np.int32)
        swap_margin = (np.array(swap_dims) / 2).astype(np.int32)
        num_sample = int((dims[1]*dims[2]*dims[3])*swap_ratio)
        output_source = source
        output_target = target

        # randomly select blind spots
        idz_msk = np.random.randint(margin[0], dims[1] - margin[0], num_sample)
        idy_msk = np.random.randint(margin[1], dims[2] - margin[1], num_sample)
        idx_msk = np.random.randint(margin[2], dims[3] - margin[2], num_sample)

        # randomly select strides for replacement
        idz_neigh = np.random.randint(-int((swap_window_dims[0]-swap_dims[0]) / 2),
                                      int((swap_window_dims[0]-swap_dims[0]) / 2), num_sample)
        idy_neigh = np.random.randint(-int((swap_window_dims[1]-swap_dims[1]) / 2),
                                      int((swap_window_dims[1]-swap_dims[1]) / 2), num_sample)
        idx_neigh = np.random.randint(-int((swap_window_dims[2]-swap_dims[2]) / 2),
                                      int((swap_window_dims[2]-swap_dims[2]) / 2), num_sample)

        # pick the neighbours
        idz_msk_neigh = idz_msk + idz_neigh
        idy_msk_neigh = idy_msk + idy_neigh
        idx_msk_neigh = idx_msk + idx_neigh

        # set boundary for swapping pixels
        idz_msk_min = idz_msk - swap_margin[0]
        idz_msk_max = idz_msk + swap_margin[0]
        idy_msk_min = idy_msk - swap_margin[1]
        idy_msk_max = idy_msk + swap_margin[1]
        idx_msk_min = idx_msk - swap_margin[2]
        idx_msk_max = idx_msk + swap_margin[2]

        idz_msk_neigh_min = idz_msk_neigh - swap_margin[0]
        idz_msk_neigh_max = idz_msk_neigh + swap_margin[0]
        idy_msk_neigh_min = idy_msk_neigh - swap_margin[1]
        idy_msk_neigh_max = idy_msk_neigh + swap_margin[1]
        idx_msk_neigh_min = idx_msk_neigh - swap_margin[2]
        idx_msk_neigh_max = idx_msk_neigh + swap_margin[2]


        # larger volume blind-spot
        if swap_region == 'volume':
            for i in range(len(idz_msk_min)):
                if swap_mode == 'internal':
                    output_source[0, idz_msk_min[i]:idz_msk_max[i]+1, idy_msk_min[i]:idy_msk_max[i]+1, idx_msk_min[i]:idx_msk_max[i]+1] = \
                        source[0, idz_msk_neigh_min[i]:idz_msk_neigh_max[i]+1, idy_msk_neigh_min[i]:idy_msk_neigh_max[i]+1, idx_msk_neigh_min[i]:idx_msk_neigh_max[i]+1]
                if swap_mode == 'cross':
                    output_source[0, idz_msk_min[i]:idz_msk_max[i]+1, idy_msk_min[i]:idy_msk_max[i]+1, idx_msk_min[i]:idx_msk_max[i]+1] = \
                        target[0, idz_msk_neigh_min[i]:idz_msk_neigh_max[i]+1, idy_msk_neigh_min[i]:idy_msk_neigh_max[i]+1, idx_msk_neigh_min[i]:idx_msk_neigh_max[i]+1]
                swap_mask[0, idz_msk_min[i]:idz_msk_max[i]+1, idy_msk_min[i]:idy_msk_max[i]+1, idx_msk_min[i]:idx_msk_max[i]+1] = 0.0
                center_mask[0, idz_msk[i], idy_msk[i], idx_msk[i]] = 1

        # single pixel blind-spot
        elif swap_region == 'single':
            for i in range(len(idz_msk_min)):
                if swap_mode == 'internal':
                   output_source[:, idz_msk[i], idy_msk[i], idx_msk[i]] = source[:, idz_msk_neigh[i], idy_msk_neigh[i], idx_msk_neigh[i]]
                if swap_mode == 'cross':
                   output_source[:, idz_msk[i], idy_msk[i], idx_msk[i]] = target[:, idz_msk[i], idy_msk[i], idx_msk[i]]

                # double masks for single blind-spot
                swap_mask[0, idz_msk[i], idy_msk[i], idx_msk[i]] = 0.0
                center_mask[0, idz_msk[i], idy_msk[i], idx_msk[i]] = 1

        # no blind-spot
        elif swap_region == 'none':
            output_source = source
            output_target = target
        #
        return output_source, output_target, swap_mask, center_mask


    def set_mode(self, mode):
        modes = ['train', 'test']
        assert mode in modes
        self.mode = mode

    def augment(self, x, y):
        # mirror axes
        for ax in range(3):
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=ax)
                y = np.flip(y, axis=ax)

        # rotate around each axis
        for ax in [(0, 1), (0, 2), (1, 2)]:
            k = np.random.randint(4)
            x = np.rot90(x, k=k, axes=ax)
            y = np.rot90(y, k=k, axes=ax)

        return np.ascontiguousarray(x), np.ascontiguousarray(y)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        input, label, mask = data['input'], data['label'], data['mask']

        input = input.astype(np.float32)
        label = label.astype(np.float32)
        mask = mask.astype(np.float32)
        return {'input': torch.from_numpy(input), 'label': torch.from_numpy(label), 'mask': torch.from_numpy(mask)}


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input, label, mask = data['input'], data['label'], data['mask']

        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std

        data = {'input': input, 'label': label, 'mask': mask}
        return data


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        input, label, mask = data['input'], data['label'], data['mask']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)
            mask = np.fliplr(mask)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            label = np.flipud(label)
            mask = np.flipud(mask)

        return {'input': input, 'label': label, 'mask': mask}


class Rescale(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label = data['input'], data['label']

    h, w = input.shape[1:]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    input = transform.resize(input, (new_h, new_w))
    label = transform.resize(label, (new_h, new_w))

    return {'input': input, 'label': label}


class RandomCrop(object):
  """Crop randomly the image in a sample volume

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    c, h, w = input.shape[:]
    new_c, new_h, new_w = self.output_size

    forward = np.random.randint(0, c - new_c)
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_c = np.arange(forward, forward + new_c, 1).astype(np.int32)
    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
    id_x = np.arange(left, left + new_w, 1).astype(np.int32)

    input = input[id_c, id_y, id_x]
    label = label[id_c, id_y, id_x]
    mask = mask[id_c, id_y, id_x]

    return {'input': input, 'label': label, 'mask': mask}


class UnifromSample(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, stride):
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
      self.stride = (stride, stride)
    else:
      assert len(stride) == 2
      self.stride = stride

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]
    stride_h, stride_w = self.stride
    new_h = h//stride_h
    new_w = w//stride_w

    top = np.random.randint(0, stride_h + (h - new_h * stride_h))
    left = np.random.randint(0, stride_w + (w - new_w * stride_w))

    id_h = np.arange(top, h, stride_h)[:, np.newaxis]
    id_w = np.arange(left, w, stride_w)

    input = input[id_h, id_w]
    label = label[id_h, id_w]
    mask = mask[id_h, id_w]

    return {'input': input, 'label': label, 'mask': mask}


class ZeroPad(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label, mask = data['input'], data['label'], data['mask']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    l = (new_w - w)//2
    r = (new_w - w) - l

    u = (new_h - h)//2
    b = (new_h - h) - u

    input = np.pad(input, pad_width=((u, b), (l, r), (0, 0)))
    label = np.pad(label, pad_width=((u, b), (l, r), (0, 0)))
    mask = np.pad(mask, pad_width=((u, b), (l, r), (0, 0)))

    return {'input': input, 'label': label, 'mask': mask}

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        return data.to('cpu').detach().numpy()

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = self.std * data + self.mean
        return data
