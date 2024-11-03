import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
from torch import nn
from sklearn import metrics
import json
import pandas as pd
from mortality_part_preprocessing import PairedDataset, MortalityDataset
from models.regular_transformer import EncoderClassifierRegular
from models.early_stopper import EarlyStopping
from models.deep_set_attention import DeepSetAttentionModel
from models.grud import GRUDModel
from models.ip_nets import InterpolationPredictionModel

plt.ion()  # interactive mode


def train_test(
    train_pair,
    val_data,
    test_data,
    output_path,
    model_type,
    model_args,
    batch_size=64,
    epochs=300,
    patience=5,
    weight_decay=0,
    lr=0.0001,
    lr_patience=1,
    early_stop_criteria="auroc"
):

    batch_size = batch_size // 2  # we concatenate 2 batches together

    train_collate_fn = PairedDataset.paired_collate_fn if model_type == "inverted" else PairedDataset.paired_collate_fn_truncate
    val_test_collate_fn = MortalityDataset.non_pair_collate_fn if model_type == "inverted" else MortalityDataset.non_pair_collate_fn_truncate


    train_dataloader = DataLoader(train_pair, batch_size, shuffle=True, num_workers=16, collate_fn=train_collate_fn, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size * 2, shuffle=True, num_workers=16, collate_fn=val_test_collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size * 2, shuffle=False, num_workers=16, collate_fn=val_test_collate_fn, pin_memory=True)

    # assign GPU
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    val_loss, model = train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_path=output_path,
        epochs=epochs,
        patience=patience,
        device=device,
        model_type=model_type,
        batch_size=batch_size,
        weight_decay=weight_decay,
        lr=lr,
        lr_patience=lr_patience,
        early_stop_criteria=early_stop_criteria,
        model_args=model_args
    )

    loss, accuracy_score, auprc_score, auc_score = test(
        test_dataloader=test_dataloader,
        output_path=output_path,
        device=device,
        model_type=model_type,
        model=model,
        model_args=model_args,
    )

    return loss, accuracy_score, auprc_score, auc_score


def only_test(
    test_data,
    output_path,
    model_type,
    model_args,
    batch_size=32,
):

    # load and create batches
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    # assign GPU
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    test_obtain_attention_weights(
        test_dataloader,
        output_path=output_path,
        device=device,
        model_type=model_type,
        model_args=model_args
    )

    # test model
    loss, accuracy_score, auprc_score, auc_score = test(
        test_dataloader,
        output_path=output_path,
        device=device,
        model_type=model_type,
        model_args=model_args
    )

    return loss, accuracy_score, auprc_score, auc_score


def train(
    train_dataloader,
    val_dataloader,
    output_path,
    epochs,
    patience,
    device,
    model_type,
    weight_decay,
    lr,
    lr_patience,
    early_stop_criteria,
    model_args,
    **kwargs,  
):
    """
    training
    """

    iterable_inner_dataloader = iter(train_dataloader)
    test_batch = next(iterable_inner_dataloader)
    max_seq_length = test_batch[0].shape[2]
    sensor_count = test_batch[0].shape[1]
    static_size = test_batch[2].shape[1]

    # make a new model and train
    if model_type == "grud":
        model = GRUDModel(
            input_dim=sensor_count,
            static_dim=static_size,
            output_dims=2,
            device=device,
            **model_args
        )
    elif model_type == "ipnets":
        model = InterpolationPredictionModel(
            output_dims=2,
            sensor_count=sensor_count,
            **model_args
        )
    elif model_type == "seft":
        model = DeepSetAttentionModel(
            output_activation=None,
            n_modalities=sensor_count,
            output_dims=2,
            **model_args
        )
    elif model_type == "vanilla":
        model = EncoderClassifierRegular(
            num_classes=2,
            device=device,
            max_timepoint_count=max_seq_length,
            sensors_count=sensor_count,
            static_count=static_size,
            return_intermediates=False,
            **model_args
        )
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"# of trainable parameters: {params}")
    criterion = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )  # weight_decay=weight_decay,
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)#, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=lr_patience,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08,
        # verbose=True, depricated
    )  # TAKEN FROM RAINDROP!

    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=f"{output_path}/checkpoint.pt"
    )  # set up early stopping

    # initialize results file
    with open(f"{output_path}/training_log.csv", "w") as train_log:
        train_log.write(
            ",".join(["epoch", "train_loss", "val_loss", "val_roc_auc_score"]) + "\n"
        )

    for epoch in range(epochs):
        # training
        model.train().to(device)  # sets training mode
        loss_list = []
        for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
            data, times, static, labels, mask, delta = batch
            if model_type != "grud":
                data = data.to(device)
                static = static.to(device)
                times = times.to(device)
                mask = mask.to(device)
                delta = delta.to(device)

            optimizer.zero_grad()

            predictions = model(
                x=data, static=static, time=times, sensor_mask=mask, delta=delta
            )
            if type(predictions) == tuple:
                predictions, recon_loss = predictions
            else:
                recon_loss = 0
            predictions = predictions.squeeze(-1)
            loss = criterion(predictions.cpu(), labels) + recon_loss
            loss_list.append(loss.item())
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        accum_loss = np.mean(loss_list)
        # validation
        model.eval().to(device)
        labels_list = torch.LongTensor([])
        predictions_list = torch.FloatTensor([])
        with torch.no_grad():
            for batch in val_dataloader:
                data, times, static, labels, mask, delta = batch
                labels_list = torch.cat((labels_list, labels), dim=0)
                if model_type != "grud":
                    data = data.to(device)
                    static = static.to(device)
                    times = times.to(device)
                    mask = mask.to(device)
                    delta = delta.to(device)
                predictions = model(
                    x=data, static=static, time=times, sensor_mask=mask, delta=delta
                )
                if type(predictions) == tuple:
                    predictions, _ = predictions
                predictions = predictions.squeeze(-1)
                predictions_list = torch.cat(
                    (predictions_list, predictions.cpu()), dim=0
                )
            probs = torch.nn.functional.softmax(predictions_list, dim=1)
            auc_score = metrics.roc_auc_score(labels_list, probs[:, 1])
            aupr_score = metrics.average_precision_score(labels_list, probs[:, 1])
            scheduler.step(aupr_score)

        val_loss = criterion(predictions_list.cpu(), labels_list)

        with open(f"{output_path}/training_log.csv", "a") as train_log:
            train_log.write(
                ",".join(map(str, [epoch + 1, accum_loss, val_loss.item(), auc_score]))
                + "\n"
            )

        print(f"Epoch: {epoch+1}, Train Loss: {accum_loss}, Val Loss: {val_loss}")
        # print(f"LR:{scheduler.get_last_lr()}")

        # set early stopping
        if early_stop_criteria == "auroc":
            early_stopping(1 - auc_score, model)
        elif early_stop_criteria == "auprc":
            early_stopping(1 - aupr_score, model)
        elif early_stop_criteria == "auprc+auroc":
            early_stopping(1 - (aupr_score + auc_score), model)
        elif early_stop_criteria == "loss":
            early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # save training curves
    training_log = pd.read_csv(f"{output_path}/training_log.csv")
    fig = plt.figure()
    fig.suptitle("training curves")
    ax0 = fig.add_subplot(121, title="loss")
    ax0.plot(training_log["train_loss"], label="Training")
    ax0.plot(training_log["val_loss"], label="Validation")
    ax0.legend()
    ax1 = fig.add_subplot(122, title="auroc")
    ax1.plot(training_log["val_roc_auc_score"], label="Training")
    ax1.legend()
    fig.savefig(f"{output_path}/train_curves.jpg")

    return val_loss, model


def test(
    test_dataloader,
    output_path,
    device,
    model_type,
    model_args,
    model=None,
    **kwargs,
):

    iterable_dataloader = iter(test_dataloader)
    test_batch = next(iterable_dataloader)
    max_seq_length = test_batch[0].shape[2]
    sensor_count = test_batch[0].shape[1]
    static_size = test_batch[2].shape[1]

    if model is None:
        if model_type == "grud":
            model = GRUDModel(
                input_dim=sensor_count,
                static_dim=static_size,
                output_dims=2,
                device=device,
                **model_args
            )
        elif model_type == "ipnets":
            model = InterpolationPredictionModel(
                output_dims=2,
                sensor_count=sensor_count,
                **model_args
            )
        elif model_type == "seft":
            model = DeepSetAttentionModel(
                output_activation=None,
                n_modalities=sensor_count,
                output_dims=2,
                **model_args
            )
        elif model_type == "vanilla":
            model = EncoderClassifierRegular(
                num_classes=2,
                device=device,
                max_timepoint_count=max_seq_length,
                sensors_count=sensor_count,
                static_count=static_size,
                return_intermediates=False,
                **model_args
            )


    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(
        torch.load(f"{output_path}/checkpoint.pt")
    )  # NEW: reload best model

    model.eval().to(device)

    labels_list = torch.LongTensor([])
    predictions_list = torch.FloatTensor([])
    with torch.no_grad():
        for batch in test_dataloader:
            data, times, static, labels, mask, delta = batch
            labels_list = torch.cat((labels_list, labels), dim=0)
            if model_type != "grud":
                data = data.to(device)
                static = static.to(device)
                times = times.to(device)
                mask = mask.to(device)
                delta = delta.to(device)
            predictions = model(
                x=data, static=static, time=times, sensor_mask=mask, delta=delta
            )
            if type(predictions) == tuple:
                predictions, _ = predictions
            predictions = predictions.squeeze(-1)
            predictions_list = torch.cat((predictions_list, predictions.cpu()), dim=0)
    loss = criterion(predictions_list.cpu(), labels_list)
    print(f"Test Loss: {loss}")

    probs = torch.nn.functional.softmax(predictions_list, dim=1)

    results = metrics.classification_report(
        labels_list, torch.argmax(probs, dim=1), output_dict=True  # predictions_list
    )
    cm = metrics.confusion_matrix(
        labels_list, torch.argmax(probs, dim=1)
    )  # predictions_list
    auc_score = metrics.roc_auc_score(labels_list, probs[:, 1])
    auprc_score = metrics.average_precision_score(labels_list, probs[:, 1])
    accuracy_score = metrics.accuracy_score(labels_list, np.argmax(probs, axis=1))
    print(results)
    print(cm)
    print(f"Accuracy = {accuracy_score}")
    print(f"AUPRC = {auprc_score}")
    print(f"AUROC = {auc_score}")

    # save test metrics
    test_metrics = {
        "test_loss": loss.item(),
        "accuracy": accuracy_score,
        "AUPRC": auprc_score,
        "AUROC": auc_score,
    }
    test_metrics.update(results)
    # test_metrics.update(cm) # TO DO: add later
    with open(f"{output_path}/test_results.json", "w") as fp:
        json.dump(test_metrics, fp)

    return loss, accuracy_score, auprc_score, auc_score


def test_obtain_attention_weights(
    test_dataloader,
    output_path,
    device,
    model_type,
    model_args,
    model=None,
    **kwargs,
):

    iterable_dataloader = iter(test_dataloader)
    test_batch = next(iterable_dataloader)
    max_seq_length = test_batch[0].shape[2]
    sensor_count = test_batch[0].shape[1]
    static_size = test_batch[2].shape[1]

    if model_type not in ("pat", "cross_pat", "vanilla", "inverted"):
        logging.warning("Can only obtain attention weights for Tranformer-style models")
        return

    if model is None:
        if model_type == "vanilla":
            model = EncoderClassifierRegular(
                num_classes=2,
                device=device,
                max_timepoint_count=max_seq_length,
                sensors_count=sensor_count,
                static_count=static_size,
                return_intermediates=True,
                **model_args
            )
    model.load_state_dict(
        torch.load(f"{output_path}/checkpoint.pt")
    )  # NEW: reload best model

    model.eval().to(device)

    labels_list = torch.LongTensor([])
    pos_agg_sensor_intermediates = None
    neg_agg_sensor_intermediates = None
    total_pos_seen = 0
    total_neg_seen = 0

    time_partitions = (
        48 * 2
    ) + 1  # 48 hours divided into half-hour segments accounting for 0

    all_pos_aggs = torch.zeros((time_partitions * time_partitions))
    counts_for_pos_aggs = torch.zeros((time_partitions * time_partitions))

    all_neg_aggs = torch.zeros((time_partitions * time_partitions))
    counts_for_neg_aggs = torch.zeros((time_partitions * time_partitions))

    all_times = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader)):
            labels = batch[3].long()
            labels_list = torch.cat((labels_list, labels), dim=0)
            data = batch[0].float().to(device)
            static = batch[2].float().to(device)
            times = batch[1].float().to(device)
            mask = batch[4].float().to(device)
            delta = batch[5].float().to(device)

            all_times.append(times.cpu())

            sensor_intermediates, time_intermediates = model(
                x=data, static=static, time=times, sensor_mask=mask, delta=delta
            )
            pos_indices = labels == 1
            neg_indices = labels == 0

            if time_intermediates is not None:
                for i in range(0, times.shape[0]):
                    times_converted = (times[i].cpu() * 2).round().long()
                    times_converted_idx1 = times_converted.repeat(max_seq_length)
                    times_converted_idx2 = torch.repeat_interleave(
                        times_converted, max_seq_length
                    )
                    attn_unrolled = time_intermediates[i].sum(0).cpu().reshape(-1)
                    counts = torch.ones_like(attn_unrolled)

                    multiplier = times_converted_idx2 * time_partitions
                    times_converted_idx1 += multiplier
                    if labels[i] == 1:
                        all_pos_aggs.scatter_reduce_(
                            0, times_converted_idx1, attn_unrolled, reduce="sum"
                        )
                        counts_for_pos_aggs.scatter_reduce_(
                            0, times_converted_idx1, counts, reduce="sum"
                        )
                    else:
                        all_neg_aggs.scatter_reduce_(
                            0, times_converted_idx1, attn_unrolled, reduce="sum"
                        )
                        counts_for_neg_aggs.scatter_reduce_(
                            0, times_converted_idx1, counts, reduce="sum"
                        )

            if sensor_intermediates is not None:
                if pos_agg_sensor_intermediates is None:
                    pos_agg_sensor_intermediates = (
                        sensor_intermediates[pos_indices].cpu().sum(dim=0)
                    )
                    neg_agg_sensor_intermediates = (
                        sensor_intermediates[neg_indices].cpu().sum(dim=0)
                    )
                else:
                    pos_agg_sensor_intermediates = (
                        pos_agg_sensor_intermediates
                        + sensor_intermediates[pos_indices].cpu().sum(dim=0)
                    )
                    neg_agg_sensor_intermediates = (
                        neg_agg_sensor_intermediates
                        + sensor_intermediates[neg_indices].cpu().sum(dim=0)
                    )
                total_pos_seen += labels[pos_indices].shape[0]
                total_neg_seen += labels[neg_indices].shape[0]

    if counts_for_pos_aggs.sum() != 0:
        pos_mean_time_intermediates = all_pos_aggs.reshape(
            (time_partitions, time_partitions)
        ) / counts_for_pos_aggs.reshape((time_partitions, time_partitions))
        neg_mean_time_intermediates = all_neg_aggs.reshape(
            (time_partitions, time_partitions)
        ) / counts_for_neg_aggs.reshape((time_partitions, time_partitions))

        fig, ax = plt.subplots()
        cax = ax.imshow(
            pos_mean_time_intermediates.numpy(), cmap="viridis", origin="lower"
        )
        fig.colorbar(cax)
        ax.set_xlabel("Time since admission (hour)", fontsize=12)
        ax.set_ylabel("Time since admission (hour)", fontsize=12)
        locs, ticks = plt.xticks()
        # plt.xticks(locs[1:-1], [x / 2 for x in [0, 20, 40, 60, 80]])
        plt.savefig(f"{output_path}/positive_mean_time_intermediates.jpg")
        plt.close()

        fig, ax = plt.subplots()
        cax = ax.imshow(
            neg_mean_time_intermediates.numpy(), cmap="viridis", origin="lower"
        )
        fig.colorbar(cax)
        ax.set_xlabel("Time since admission (hour)", fontsize=12)
        ax.set_ylabel("Time since admission (hour)", fontsize=12)
        locs, ticks = plt.xticks()
        # plt.xticks(locs[1:-1], [x / 2 for x in [0, 20, 40, 60, 80]])
        plt.savefig(f"{output_path}/negative_mean_time_intermediates.jpg")
        plt.close()

    if pos_agg_sensor_intermediates is not None:
        if len(pos_agg_sensor_intermediates.shape) == 3:
            pos_agg_sensor_intermediates = pos_agg_sensor_intermediates.sum(0)
            neg_agg_sensor_intermediates = neg_agg_sensor_intermediates.sum(0)
        pos_mean_sensor_intermediates = (
            pos_agg_sensor_intermediates / total_pos_seen
        ).squeeze(0)
        neg_mean_sensor_intermediates = (
            neg_agg_sensor_intermediates / total_neg_seen
        ).squeeze(0)

        fig, ax = plt.subplots()
        cax = ax.imshow(
            pos_mean_sensor_intermediates.numpy(), cmap="viridis", origin="lower"
        )
        fig.colorbar(cax)
        ax.set_xlabel("Sensor Id", fontsize=12)
        ax.set_ylabel("Sensor Id", fontsize=12)
        plt.savefig(f"{output_path}/positive_mean_sensor_intermediates.jpg")
        plt.close()

        fig, ax = plt.subplots()
        cax = ax.imshow(
            neg_mean_sensor_intermediates.numpy(), cmap="viridis", origin="lower"
        )
        fig.colorbar(cax)
        ax.set_xlabel("Sensor Id", fontsize=12)
        ax.set_ylabel("Sensor Id", fontsize=12)
        plt.savefig(f"{output_path}/negative_mean_sensor_intermediates.jpg")
        plt.close()

    return