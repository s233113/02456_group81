import json
import h5py
import os
import numpy as np
import tqdm
import torch
from torch.utils.data import Dataset


def truncate_to_longest_item_in_batch(data, times, mask, delta):
    data = data.permute((0, 2, 1))  # (N, T, F)
    mask = mask.permute((0, 2, 1))
    delta = delta.permute((0, 2, 1))
    col_mask = mask.sum(-1)
    valid_time_points = col_mask.any(dim=0)
    data = data[:, valid_time_points, :].permute((0, 2, 1))
    times = times[:, valid_time_points]
    mask = mask[:, valid_time_points, :].permute((0, 2, 1))
    delta = delta[:, valid_time_points, :].permute((0, 2, 1))
    return data, times, mask, delta


def load_pad_separate(dataset_id, base_path="", split_index=1, save_path="./processed_datasets"):
    """
    loads, zero pads, and separates data preprocessed by SeFT

    files structured as dict = = [{
                "ts_values": normalized_values[i],
                "ts_indicators": normalized_measurements[i],
                "ts_times": normalized_times[i],
                "static": normalized_static[i],
                "labels": normalized_labels[i]}]
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # File paths for preprocessed datasets
    pos_path = os.path.join(save_path, f"{dataset_id}_{split_index}_pos.h5")
    neg_path = os.path.join(save_path, f"{dataset_id}_{split_index}_neg.h5")
    val_path = os.path.join(save_path, f"{dataset_id}_{split_index}_val.h5")
    test_path = os.path.join(save_path, f"{dataset_id}_{split_index}_test.h5")

    # Check if the preprocessed files already exist, and load them if they do
    if save_path and all(os.path.exists(p) for p in [pos_path, neg_path, val_path, test_path]):
        print(f"Loading preprocessed datasets from {save_path}")
        mortality_pos = MortalityDataset(hdf5_path=pos_path)
        mortality_neg = MortalityDataset(hdf5_path=neg_path)
        mortality_val = MortalityDataset(hdf5_path=val_path)
        mortality_test = MortalityDataset(hdf5_path=test_path)
    else:
        # If preprocessed files are not available, proceed with preprocessing
        print(f"Preprocessed files not found. Preprocessing the dataset...")
        Ptrain, Pval, Ptest, norm_params = dataset_loader_splitter(
            dataset_id, base_path, split_index
        )

        # Determine max length based on dataset
        if dataset_id == "physionet2012":
            max_len = 215
        else:
            raise ValueError(f"Dataset {dataset_id} not recognised")

        # Preprocess the datasets
        mortality_pos = MortalityDataset(
            Ptrain, max_length=max_len, norm_params=norm_params
        )
        mortality_neg = MortalityDataset(
            Ptrain, max_length=max_len, norm_params=norm_params
        )
        mortality_val = MortalityDataset(Pval, max_length=max_len, norm_params=norm_params)
        mortality_test = MortalityDataset(
            Ptest, max_length=max_len, norm_params=norm_params
        )

        # separate pos v neg samples for equal class representation in batches
        ytrain = [item.get("labels") for item in Ptrain]
        ytrain = np.array(ytrain)
        nonzeroes = ytrain.nonzero()[0]
        zeroes = np.where(ytrain == 0)[0]

        # we separate the positive and negative datasets so that we can upsample
        mortality_pos.select_indices(nonzeroes)
        mortality_neg.select_indices(zeroes)

        # Save the preprocessed datasets if save_path is provided
        if save_path:
            print(f"Saving datasets to {save_path}")
            mortality_pos.save_to_hdf5(pos_path)
            mortality_neg.save_to_hdf5(neg_path)
            mortality_val.save_to_hdf5(val_path)
            mortality_test.save_to_hdf5(test_path)

    mortality_pair = PairedDataset(mortality_pos, mortality_neg)

    return mortality_pair, mortality_val, mortality_test


def dataset_loader_splitter(dataset_id, base_path, split_index):
    """loads and splits data"""

    split_path_train = "/train_" + dataset_id + "_" + str(split_index) + ".npy"
    split_path_val = "/validation_" + dataset_id + "_" + str(split_index) + ".npy"
    split_path_test = "/test_" + dataset_id + "_" + str(split_index) + ".npy"
    split_path_norm = "/normalization_" + dataset_id + "_" + str(split_index) + ".json"

    print("Loading dataset")

    # extract train/val/test obs and labels
    Ptrain = np.load(base_path + split_path_train, allow_pickle=True)
    Pval = np.load(base_path + split_path_val, allow_pickle=True)
    Ptest = np.load(base_path + split_path_test, allow_pickle=True)
    try:
        norm_params = json.load(open(base_path + split_path_norm))
    except Exception:
        norm_params = None

    return Ptrain, Pval, Ptest, norm_params


class PairedDataset(Dataset):

    def __init__(self, dataset_pos, dataset_neg, neg_sample=False):
        self.dataset_pos = dataset_pos
        self.dataset_neg = dataset_neg
        self.neg_sample = neg_sample
        if not self.neg_sample:
            self.dataset_pos.repeat_data(3)

    def __len__(self):
        if self.neg_sample:
            return len(self.dataset_neg)
        else:
            return len(self.dataset_pos)

    def _getitem_negative(self, idx):
        pos_data = self.dataset_pos[idx % len(self.dataset_pos)]
        neg_data = self.dataset_neg[idx]
        return pos_data, neg_data

    def _getitem_positive(self, idx):
        pos_data = self.dataset_pos[idx]
        neg_data = self.dataset_neg[idx % len(self.dataset_neg)]
        return pos_data, neg_data

    def __getitem__(self, idx):
        return self._getitem_negative(idx) if self.neg_sample else self._getitem_positive(idx)

    @staticmethod
    def paired_collate_fn(batch):
        """
        Custom collate function to concatenate and shuffle the paired positive and negative batches.
        """
        # Unzip the batch into two lists: positive and negative batches
        pos_batch, neg_batch = zip(*batch)

        # Extract individual elements (data, labels, etc.) from both batches
        pos_data, pos_times, pos_static, pos_labels, pos_mask, pos_delta = zip(*pos_batch)
        neg_data, neg_times, neg_static, neg_labels, neg_mask, neg_delta = zip(*neg_batch)

        # Concatenate each element (data, labels, etc.)
        data = torch.stack(pos_data + neg_data)
        times = torch.stack(pos_times + neg_times)
        static = torch.stack(pos_static + neg_static)
        labels = torch.stack(pos_labels + neg_labels)
        mask = torch.stack(pos_mask + neg_mask)
        delta = torch.stack(pos_delta + neg_delta)

        # Create a list of indices for shuffling
        indices = torch.randperm(data.size(0))

        # Shuffle the concatenated tensors based on the random indices
        data = data[indices]
        times = times[indices]
        static = static[indices]
        labels = labels[indices]
        mask = mask[indices]
        delta = delta[indices]

        return data, times, static, labels, mask, delta
    
    @staticmethod
    def paired_collate_fn_truncate(batch):
        data, times, static, labels, mask, delta = PairedDataset.paired_collate_fn(batch)
        data, times, mask, delta = truncate_to_longest_item_in_batch(data, times, mask, delta)
        return data, times, static, labels, mask, delta


class MortalityDataset(Dataset):

    def __init__(self, obs=None, max_length=2881, norm_params=None, hdf5_path=None):
        """
        Arguments:
            obs: all experimental results, including active sensors, static sensors, and times (as dict)
        """
        if hdf5_path:
            # Load the dataset from an HDF5 file
            self.load_from_hdf5(hdf5_path)
        else:
            # Process the data if raw observations are provided
            self.norm_params = norm_params
            print("Preprocessing dataset")
            (
                self.data_array,
                self.sensor_mask_array,
                self.times_array,
                self.static_array,
                self.label_array,
                self.delta_array,
            ) = MortalityDataset.preprocess_sensor_readings(max_length, obs)
            self.data_array = self.data_array.permute((0, 2, 1))
            self.sensor_mask_array = self.sensor_mask_array.permute((0, 2, 1))
            self.delta_array = self.delta_array.permute((0, 2, 1))
            print("shape of active data = " + str(np.shape(self.data_array)))
            print("shape of time data = " + str(np.shape(self.times_array)))
            print("shape of static data = " + str(np.shape(self.static_array)))

    def save_to_hdf5(self, hdf5_path):
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('data_array', data=self.data_array)
            f.create_dataset('sensor_mask_array', data=self.sensor_mask_array)
            f.create_dataset('times_array', data=self.times_array)
            f.create_dataset('static_array', data=self.static_array)
            f.create_dataset('label_array', data=self.label_array)
            f.create_dataset('delta_array', data=self.delta_array)
            # Save norm_params as JSON string (since it's a dict)
            f.attrs['norm_params'] = json.dumps(self.norm_params)

    # Load the dataset from HDF5 file
    def load_from_hdf5(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            self.data_array = torch.tensor(f['data_array'][:], dtype=torch.float32)
            self.sensor_mask_array = torch.tensor(f['sensor_mask_array'][:], dtype=torch.float32)
            self.times_array = torch.tensor(f['times_array'][:], dtype=torch.float32)
            self.static_array = torch.tensor(f['static_array'][:], dtype=torch.float32)
            self.label_array = torch.tensor(f['label_array'][:], dtype=torch.long)
            self.delta_array = torch.tensor(f['delta_array'][:], dtype=torch.float32)
            self.norm_params = json.loads(f.attrs['norm_params'])
        print(f"Loaded dataset from {hdf5_path}")

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        return (
            self.data_array[idx],
            self.times_array[idx],
            self.static_array[idx],
            self.label_array[idx],
            self.sensor_mask_array[idx],
            self.delta_array[idx],
        )

    def select_indices(self, indices):
        self.data_array = self.data_array[indices]
        self.times_array = self.times_array[indices]
        self.static_array = self.static_array[indices]
        self.label_array = self.label_array[indices]
        self.sensor_mask_array = self.sensor_mask_array[indices]
        self.delta_array = self.delta_array[indices]
        print("shape of active data = " + str(np.shape(self.data_array)))
        print("shape of time data = " + str(np.shape(self.times_array)))
        print("shape of static data = " + str(np.shape(self.static_array)))
        print("shape of labels = " + str(np.shape(self.label_array)))

    def repeat_data(self, n):
        self.data_array = self.data_array.repeat(n, 1, 1)
        self.times_array = self.times_array.repeat(n, 1)
        self.static_array = self.static_array.repeat(n, 1)
        self.label_array = self.label_array.repeat(n)
        self.sensor_mask_array = self.sensor_mask_array.repeat(n, 1, 1)
        self.delta_array = self.delta_array.repeat(n, 1, 1)

    @staticmethod
    def preprocess_sensor_readings(max_length, dict_set):
        """
        turn mimic into an array,
        of dimension (times, sensor_mask, subjects, obs)
        with missing readings as zero
        """
        # make a list to hold all individuals
        data_list = []
        sensor_mask_list = []
        static_list = []
        times_list = []
        labels_list = []
        delta_list = []

        # for each individual,
        for ind in tqdm.tqdm(dict_set):
            # get times, obs values, and static obs values
            times = ind.get("ts_times")
            sensor_mask = ind.get("ts_indicators")
            obs = ind.get("ts_values")  # this is readings for the 36 sensors
            stat = ind.get(
                "static"
            )  # this is static readings for the 9 static data types
            label = ind.get("labels")
            label = np.amax(label)

            # get size of times list
            if len(times) < max_length:
                # zero pad the time list
                padding_zeros_times = max_length - len(times)
                times = np.pad(
                    times, (0, padding_zeros_times), "constant", constant_values=(0.0)
                )
                # zero pad the observations list
                padding_zeros_obs = np.full(
                    (padding_zeros_times, obs.shape[1]), 0, dtype=float
                )
                obs = np.append(obs, padding_zeros_obs, axis=0)
                # zero pad the sensors mask CHECK IS RIGHT
                padding_zeros_mask = np.full(
                    (padding_zeros_times, obs.shape[1]), 0, dtype=bool
                )
                sensor_mask = np.append(sensor_mask, padding_zeros_mask, axis=0)

            # create array of time delta since last reading
            delta = get_delta_t(times, obs, sensor_mask)  # (T, F)
            data_list.append(obs)
            sensor_mask_list.append(sensor_mask)
            times_list.append(times)
            static_list.append(stat)
            labels_list.append(label)
            delta_list.append(delta)

        data_array = np.stack(data_list)
        sensor_mask_array = np.stack(sensor_mask_list)
        time_array = np.stack(times_list)
        static_array = np.stack(static_list)
        label_array = np.stack(labels_list)
        delta_array = np.stack(delta_list)

        return (
            torch.tensor(data_array, dtype=torch.float32),
            torch.tensor(sensor_mask_array, dtype=torch.float32),
            torch.tensor(time_array, dtype=torch.float32),
            torch.tensor(static_array, dtype=torch.float32),
            torch.tensor(label_array, dtype=torch.long),
            torch.tensor(delta_array, dtype=torch.float32),
        )

    @staticmethod
    def non_pair_collate_fn(batch):
        """
        Custom collate function for the validation dataloader.
        This function organizes the batch into (data, times, static, labels, mask, delta).
        """
        data, times, static, labels, mask, delta = zip(*batch)

        data = torch.stack(data).float()
        times = torch.stack(times).float()
        static = torch.stack(static).float()
        labels = torch.stack(labels).long()
        mask = torch.stack(mask).float()
        delta = torch.stack(delta).float()

        return data, times, static, labels, mask, delta
    
    @staticmethod
    def non_pair_collate_fn_truncate(batch):
        data, times, static, labels, mask, delta = MortalityDataset.non_pair_collate_fn(batch)
        data, times, mask, delta = truncate_to_longest_item_in_batch(data, times, mask, delta)
        return data, times, static, labels, mask, delta


def get_delta_t(times, measurements, measurement_indicators):
    """
    Modified from SeFT's GRU-D Implementation.

    Creates array with time from most recent feature measurement.
    """
    dt_list = []

    # First observation has dt = 0
    first_dt = np.zeros(measurement_indicators.shape[1:], dtype=np.float32)  # (F,)
    dt_list.append(first_dt)

    last_dt = first_dt.copy()  # Initialize last_dt before the loop
    for i in range(1, measurement_indicators.shape[0]):
        last_dt = np.where(
            measurement_indicators[i - 1],
            np.full_like(last_dt, times[i] - times[i - 1]),
            times[i] - times[i - 1] + last_dt,
        )
        dt_list.append(last_dt)

    dt_array = np.stack(dt_list)
    dt_array = dt_array.astype(np.float32)  # Ensure consistent data type
    dt_array.shape = measurements.shape  # Reshape to match measurements
    dt_array = dt_array * ~(measurement_indicators.astype(bool))

    return dt_array
