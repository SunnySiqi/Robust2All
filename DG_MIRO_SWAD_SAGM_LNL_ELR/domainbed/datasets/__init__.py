import torch
import numpy as np

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT

# Add asym noise to DomainNet
asym_noise_dict = {0: 308,
 1: 208,
 2: 28,
 3: 135,
 4: 5,
 5: 0,
 6: 0,
 7: 324,
 8: 324,
 9: 208,
 10: 288,
 11: 324,
 12: 208,
 13: 285,
 14: 208,
 15: 16,
 16: 17,
 17: 282,
 18: 19,
 19: 327,
 20: 309,
 21: 208,
 22: 327,
 23: 208,
 24: 288,
 25: 135,
 26: 27,
 27: 28,
 28: 208,
 29: 208,
 30: 327,
 31: 98,
 32: 33,
 33: 144,
 34: 35,
 35: 308,
 36: 282,
 37: 38,
 38: 327,
 39: 208,
 40: 208,
 41: 42,
 42: 208,
 43: 44,
 44: 308,
 45: 46,
 46: 331,
 47: 324,
 48: 91,
 49: 90,
 50: 327,
 51: 324,
 52: 53,
 53: 324,
 54: 327,
 55: 331,
 56: 282,
 57: 151,
 58: 334,
 59: 324,
 60: 324,
 61: 208,
 62: 175,
 63: 64,
 64: 327,
 65: 208,
 66: 67,
 67: 68,
 68: 208,
 69: 208,
 70: 138,
 71: 331,
 72: 324,
 73: 175,
 74: 53,
 75: 254,
 76: 338,
 77: 276,
 78: 91,
 79: 208,
 80: 282,
 81: 208,
 82: 282,
 83: 319,
 84: 85,
 85: 208,
 86: 310,
 87: 324,
 88: 208,
 89: 90,
 90: 91,
 91: 208,
 92: 323,
 93: 285,
 94: 95,
 95: 261,
 96: 276,
 97: 98,
 98: 324,
 99: 282,
 100: 288,
 101: 102,
 102: 103,
 103: 327,
 104: 110,
 105: 288,
 106: 107,
 107: 282,
 108: 276,
 109: 110,
 110: 324,
 111: 110,
 112: 288,
 113: 114,
 114: 157,
 115: 208,
 116: 327,
 117: 98,
 118: 327,
 119: 208,
 120: 208,
 121: 110,
 122: 324,
 123: 208,
 124: 125,
 125: 208,
 126: 208,
 127: 324,
 128: 129,
 129: 208,
 130: 327,
 131: 208,
 132: 208,
 133: 28,
 134: 135,
 135: 136,
 136: 324,
 137: 138,
 138: 35,
 139: 282,
 140: 324,
 141: 208,
 142: 208,
 143: 282,
 144: 324,
 145: 146,
 146: 282,
 147: 148,
 148: 208,
 149: 208,
 150: 151,
 151: 98,
 152: 153,
 153: 308,
 154: 208,
 155: 341,
 156: 157,
 157: 208,
 158: 324,
 159: 208,
 160: 208,
 161: 98,
 162: 163,
 163: 208,
 164: 282,
 165: 308,
 166: 230,
 167: 1,
 168: 285,
 169: 208,
 170: 171,
 171: 208,
 172: 208,
 173: 208,
 174: 175,
 175: 208,
 176: 282,
 177: 178,
 178: 110,
 179: 246,
 180: 208,
 181: 282,
 182: 324,
 183: 282,
 184: 208,
 185: 324,
 186: 324,
 187: 188,
 188: 282,
 189: 190,
 190: 324,
 191: 282,
 192: 193,
 193: 135,
 194: 35,
 195: 28,
 196: 282,
 197: 307,
 198: 178,
 199: 208,
 200: 208,
 201: 28,
 202: 324,
 203: 282,
 204: 208,
 205: 206,
 206: 282,
 207: 208,
 208: 91,
 209: 324,
 210: 211,
 211: 212,
 212: 213,
 213: 288,
 214: 208,
 215: 216,
 216: 282,
 217: 246,
 218: 335,
 219: 276,
 220: 282,
 221: 222,
 222: 208,
 223: 327,
 224: 110,
 225: 285,
 226: 208,
 227: 228,
 228: 208,
 229: 324,
 230: 327,
 231: 232,
 232: 208,
 233: 282,
 234: 282,
 235: 324,
 236: 327,
 237: 208,
 238: 285,
 239: 240,
 240: 331,
 241: 285,
 242: 324,
 243: 208,
 244: 309,
 245: 107,
 246: 247,
 247: 248,
 248: 324,
 249: 321,
 250: 251,
 251: 288,
 252: 135,
 253: 254,
 254: 327,
 255: 208,
 256: 208,
 257: 341,
 258: 208,
 259: 135,
 260: 261,
 261: 262,
 262: 208,
 263: 213,
 264: 208,
 265: 327,
 266: 208,
 267: 268,
 268: 269,
 269: 208,
 270: 309,
 271: 208,
 272: 273,
 273: 135,
 274: 208,
 275: 276,
 276: 277,
 277: 324,
 278: 279,
 279: 208,
 280: 281,
 281: 282,
 282: 282,
 283: 208,
 284: 285,
 285: 98,
 286: 282,
 287: 208,
 288: 310,
 289: 324,
 290: 282,
 291: 309,
 292: 208,
 293: 294,
 294: 208,
 295: 324,
 296: 327,
 297: 208,
 298: 208,
 299: 324,
 300: 208,
 301: 285,
 302: 324,
 303: 282,
 304: 282,
 305: 282,
 306: 307,
 307: 308,
 308: 282,
 309: 282,
 310: 341,
 311: 208,
 312: 313,
 313: 331,
 314: 282,
 315: 282,
 316: 282,
 317: 282,
 318: 282,
 319: 327,
 320: 327,
 321: 282,
 322: 208,
 323: 324,
 324: 325,
 325: 324,
 326: 327,
 327: 282,
 328: 329,
 329: 282,
 330: 282,
 331: 332,
 332: 324,
 333: 282,
 334: 335,
 335: 208,
 336: 337,
 337: 338,
 338: 208,
 339: 340,
 340: 341,
 341: 342,
 342: 208,
 343: 344,
 344: 282}
def set_transfroms(dset, data_type, hparams, algorithm_class=None):
    """
    Args:
        data_type: ['train', 'valid', 'test', 'mnist', 'CP_train', 'CP_valid', 'CP_test']
    """
    assert hparams["data_augmentation"]

    additional_data = False
    if data_type == "train":
        dset.transforms = {"x": DBT.aug}
        #dset.transforms = {"x1": DBT.basic, "x2": DBT.aug}
        additional_data = True
    elif data_type == "valid":
        if hparams["val_augment"] is False:
            dset.transforms = {"x": DBT.basic}
        else:
            # Originally, DomainBed use same training augmentation policy to validation.
            # We turn off the augmentation for validation as default,
            # but left the option to reproducibility.
            dset.transforms = {"x1": DBT.basic, "x2": DBT.aug}
    elif data_type == "test":
        dset.transforms = {"x": DBT.basic}
    elif data_type == "mnist":
        # No augmentation for mnist
        dset.transforms = {"x": lambda x: x}
    elif data_type == 'CP_train':
        dset.transforms = {"x": DBT.CP_strong}
    elif data_type == 'CP_valid' or data_type == 'CP_test':
        dset.transforms = {"x": DBT.CP_test}
    else:
        raise ValueError(data_type)

    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform


def get_dataset(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed).
        # It means that the split is always identical only if use same trial_seed,
        # independent to run the code where, when, or how many times.
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
            env_i in test_envs,
            noise_ratio= args.noise_ratio, 
            noise_type=args.noise_type, 
            add_noise=args.add_noise
        )
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"

        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"
        
        if 'CP' in args.dataset:
            in_type = "CP_" + in_type
            out_type = "CP_" + out_type

        set_transfroms(in_, in_type, hparams, algorithm_class)
        set_transfroms(out, out_type, hparams, algorithm_class)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    return dataset, in_splits, out_splits


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys, test=False, noise_ratio=0, noise_type='sym', add_noise=False, in_type=False):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}
        self.test = test
        self.in_type = in_type
        self.direct_return = isinstance(underlying_dataset, _SplitDataset)
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type
        self.add_noise = add_noise

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        clean_y = y
        if self.add_noise and not self.test and self.in_type and np.random.uniform() < self.noise_ratio:
            y = asym_noise_dict[clean_y]
        ret = {"y": y}
        ret["key"] = key

        for key, transform in self.transforms.items():
            ret[key] = transform(x)

        return ret

    def __len__(self):
        return len(self.keys)
    
def split_dataset(dataset, n, seed=0, test=False, noise_ratio=0, noise_type='sym', add_noise=False):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1, test, noise_ratio=noise_ratio, noise_type=noise_type, add_noise=add_noise, in_type=False), _SplitDataset(dataset, keys_2, test, noise_ratio=noise_ratio, noise_type=noise_type, add_noise=add_noise, in_type=True)
