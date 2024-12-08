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
import os
from mortality_part_preprocessing import PairedDataset, MortalityDataset
from models.model_try import MambaFinetune
from transformers.models.mamba.modeling_mamba import MambaForCausalLM
from models.regular_transformer import EncoderClassifierRegular
from models.early_stopper import EarlyStopping
from models.deep_set_attention import DeepSetAttentionModel
from models.grud import GRUDModel
from models.ip_nets import InterpolationPredictionModel
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Tuple
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
    lr=0.0001,
    early_stop_criteria="auroc"
):
    torch.cuda.empty_cache()

    train_batch_size = batch_size // 2  # we concatenate 2 batches together

    train_collate_fn = PairedDataset.paired_collate_fn_truncate
    val_test_collate_fn = MortalityDataset.non_pair_collate_fn_truncate

    train_dataloader = DataLoader(train_pair, train_batch_size, shuffle=True, num_workers=16, collate_fn=train_collate_fn, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, num_workers=16, collate_fn=val_test_collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size, shuffle=False, num_workers=16, collate_fn=val_test_collate_fn, pin_memory=True)
    
    # assign GPU
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    val_loss, model = train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_path=output_path,
        epochs=epochs,
        patience=patience,
        device=device,
        model_type=model_type,
        batch_size=batch_size,
        lr=lr,
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


def resize_weights(original_weights: dict):
    hiddensize=768
    d_inner=256
    vocab=10000

    size_changes = {
        torch.Size([50280, 768]):torch.Size([vocab, hiddensize]),
        torch.Size([768]): torch.Size([hiddensize]),
        torch.Size([1536, 16]): torch.Size([d_inner, 16]),
        torch.Size([1536]): torch.Size([d_inner]),
        torch.Size([1536, 1, 4]): torch.Size([d_inner, 1, 4]),
        torch.Size([3072, 768]): torch.Size([d_inner*2, hiddensize]),
        torch.Size([80, 1536]): torch.Size([80, d_inner]),
        torch.Size([1536, 48]): torch.Size([d_inner, 48]),
        torch.Size([768, 1536]): torch.Size([hiddensize, d_inner]),
        # Add any other size changes here as needed
    }

    resized_weights = {}

    # Iterate through the original weights and resize them if needed
    for name, param in original_weights.items():
        for old_size, new_size in size_changes.items():
            if param.size() == old_size:
                print(f"Resizing {name}: {param.size()} -> {new_size}")
                # Resize the parameter to the new size (trimming or padding)
                resized_weights[name] = param.data[:new_size[0], :new_size[1]] if len(new_size) > 1 else param.data[:new_size[0]]
                break
        else:
            resized_weights[name] = param

    return resized_weights

def train(
    train_dataloader,
    val_dataloader,
    output_path,
    epochs,
    patience,
    device,
    model_type,
    lr,
    early_stop_criteria,
    model_args,
    **kwargs,
):
    """
    Training function with support for multiple model types, including Mamba.
    """
    iterable_inner_dataloader = iter(train_dataloader)
    test_batch = next(iterable_inner_dataloader)
    max_seq_length = test_batch[0].shape[2]
    sensor_count = test_batch[0].shape[1]
    static_size = test_batch[2].shape[1]
    batch_size=  test_batch[0].shape[0]

    # Initialize the model
    if model_type == "mamba":
        
        #Load pretrained mamba model and save its weights
        weights_model = AutoModelForCausalLM.from_pretrained("whaleloops/clinicalmamba-130m-hf")
        original_weights = weights_model.state_dict()
        # Resize the weights before loading them into the model
        resized_weights = resize_weights(original_weights)

        #Load config and edit it to reduce layers and vocab size
        config = AutoConfig.from_pretrained("whaleloops/clinicalmamba-130m-hf")

        # Update the configuration with reduced dimensions
        config.hidden_size = 768
        config.d_model = 768
        config.intermediate_size = 256
        config.d_inner = 256
        config.vocab_size= 10000
        #config.torch_dtype = "float16"

        #Match our new mamba architecture with the original (compatible) weight
        #resize embedding layer
        pretrained_model= AutoModelForCausalLM.from_config(config)
        # Now load the adjusted weights into the model (strict=False allows for mismatches)
        pretrained_model.load_state_dict(resized_weights, strict=False)
  
        model = MambaFinetune(
            pretrained_model=pretrained_model,
            train_data=train_dataloader.dataset,
            train_data_loader=train_dataloader,
            problem_type="single_label_classification",
            num_labels=2,
            vocab_size=model_args.get("vocab_size", 9),
            learning_rate=lr,
            classifier_dropout=model_args.get("dropout", 0.1),
        )

        model=model.to(device)

    elif model_type == "grud":
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
    elif model_type == "transformer":
        model = EncoderClassifierRegular(
            num_classes=2,
            device=device,
            max_timepoint_count=max_seq_length,
            sensors_count=sensor_count,
            static_count=static_size,
            return_intermediates=False,
            **model_args
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"# of trainable parameters: {params}")

    criterion = nn.CrossEntropyLoss()
    print("CrossEntropyLoss")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Adam")
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=f"{output_path}/checkpoint.pt"
    )

    # Initialize results file
    with open(f"{output_path}/training_log.csv", "w") as train_log:
        train_log.write(
            ",".join(["epoch", "train_loss", "val_loss", "val_roc_auc_score"]) + "\n"
        )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) #extra additions

    #try to pass model to device outside for loop

    #model=model.to(device)
    
    for epoch in range(epochs):
        
        #.to(device) # CHECK WHAT THIS LINE IS DOING
        #We empty the cache before training
        torch.cuda.empty_cache()
        model.train()
        
        
        loss_list = []

        for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
            data, times, static, labels, mask, delta = batch

            if (data.shape[0]==batch_size):
                if model_type != "grud" :
                    data, static, times, mask, delta = (
                        data.to(device), static.to(device), times.to(device),
                        mask.to(device), delta.to(device)
                    )
                    if model_type=="mamba":
                        labels=labels.to(device)

                optimizer.zero_grad()

                #Call the model differently depending on if it's mamba

                if model_type == "mamba":
                    input_mamba: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (data, mask, times, static)

                    # input_mamba=Tuple[data, mask, times, static]
                    _, logits = model(inputs=input_mamba, labels=labels) #we dont use this loss, we compute it with rachel's methods
                    predictions = torch.argmax(logits, dim=1)
                    #print("shape logits train: ", logits.shape)

                    print("Labels and logits shape: ", logits.shape, labels.shape)
                else:
                    predictions = model(
                        x=data, static=static, time=times, sensor_mask=mask, delta=delta
                    )


                if type(predictions) == tuple:
                    predictions, recon_loss = predictions
                else:
                    recon_loss = 0
                
                if model_type!="mamba":
                    predictions = predictions.squeeze(-1)

                if model_type == "mamba":
                    #loss = torch.tensor(criterion(logits.squeeze(-1), labels) + recon_loss, requires_grad=True)
                    loss = criterion(logits.squeeze(-1), labels) + recon_loss
                else:
                    loss = torch.tensor(criterion(predictions.cpu(), labels) + recon_loss, requires_grad=False)


                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #extra additions
                optimizer.step()
                print("loss in train: ", loss)
                loss_list.append(loss.item())

            accum_loss = np.mean(loss_list)
            
        #We empty the cache after training
        torch.cuda.empty_cache()

        # Validation
        #model.eval().to(device)
        model.eval()
        labels_list, predictions_list, logits_list = torch.LongTensor([]), torch.FloatTensor([]), torch.FloatTensor([])
        with torch.no_grad():
            for batch in val_dataloader:
                data, times, static, labels, mask, delta = batch

                if data.shape[0]==batch_size:
                    labels_list = torch.cat((labels_list, labels), dim=0)
                    if model_type != "grud":
                        data, static, times, mask, delta = (
                            data.to(device), static.to(device), times.to(device),
                            mask.to(device), delta.to(device)
                        )
                    
                    if model_type == "mamba":
                        labels=labels.to(device)
                        input_mamba: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (data, mask, times, static)
                        #model = model.to(device)
                        _, logits = model(inputs=input_mamba, labels=labels)
                        predictions = torch.argmax(logits, dim=1)
                    else:
                        predictions = model(
                            x=data, static=static, time=times, sensor_mask=mask, delta=delta
                        )
                    
                    if type(predictions) == tuple:
                        predictions, _ = predictions

                    if model_type=="mamba":
                        logits_list = torch.cat((logits_list, logits.squeeze(-1).cpu()), dim=0)
                        #print("logits list: " , logits_list)
                    else:
                        predictions = predictions.squeeze(-1)
                        predictions_list = torch.cat((predictions_list, predictions.cpu()), dim=0)


            if model_type=="mamba":
                probs = torch.nn.functional.softmax(logits_list, dim=1)
                auc_score = metrics.roc_auc_score(labels_list, probs[:, 1])
                aupr_score = metrics.average_precision_score(labels_list, probs[:, 1])
            else:
                probs = torch.nn.functional.softmax(predictions_list, dim=1)
                auc_score = metrics.roc_auc_score(labels_list, probs[:, 1])
                aupr_score = metrics.average_precision_score(labels_list, probs[:, 1])

        if model_type=="mamba":
            val_loss=criterion(logits_list.cpu(),labels_list)
        else:
            val_loss = torch.tensor(criterion(predictions_list.cpu(), labels_list), requires_grad=False)

        # Log and early stopping
        with open(f"{output_path}/training_log.csv", "a") as train_log:
            train_log.write(
                ",".join(map(str, [epoch + 1, accum_loss, val_loss.item(), auc_score]))
                + "\n"
            )

        print(f"Epoch {epoch+1}: Train Loss = {accum_loss}, Val Loss = {val_loss}")

        scheduler.step(val_loss)
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


def test(test_dataloader, output_path, device, model_type, model, **kwargs):
    """
    Testing function for evaluating the model performance.
    """
    
    #We empty the cache before testing
    torch.cuda.empty_cache()
    
    iterable_dataloader = iter(test_dataloader)
    test_batch = next(iterable_dataloader)
    max_seq_length = test_batch[0].shape[2]
    sensor_count = test_batch[0].shape[1]
    static_size = test_batch[2].shape[1]
    batch_size=  test_batch[0].shape[0]

    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(f"{output_path}/checkpoint.pt"))
    model.eval() #.to(device)

    labels_list, predictions_list, logits_list = torch.LongTensor([]), torch.FloatTensor([]), torch.FloatTensor([])

    with torch.no_grad():
        for batch in test_dataloader:
            data, times, static, labels, mask, delta = batch
            if data.shape[0]==batch_size:
                labels_list = torch.cat((labels_list, labels), dim=0)
                if model_type != "grud":
                    data, static, times, mask, delta = (
                        data.to(device), static.to(device), times.to(device),
                        mask.to(device), delta.to(device)
                    )
                    
                if model_type == "mamba":
                    labels=labels.to(device)
                    input_mamba: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (data, mask, times, static)
                    _, logits = model(inputs=input_mamba, labels=labels)
                    predictions = torch.argmax(logits, dim=1)
                else:
                    predictions = model(
                        x=data, static=static, time=times, sensor_mask=mask, delta=delta
                    )
                if type(predictions) == tuple:
                    predictions, _ = predictions
                
                
                predictions = predictions.squeeze(-1)
                predictions_list = torch.cat((predictions_list, predictions.cpu()), dim=0)
                logits_list = torch.cat((logits_list, logits.cpu()), dim=0)
    if model_type!="mamba":
        loss = criterion(predictions_list.cpu(), labels_list)
    else:
        loss = criterion(logits_list.cpu(), labels_list)
    print(f"Test Loss: {loss}")

    if model_type=="mamba":
        probs = torch.nn.functional.softmax(logits_list, dim=1)
    else:
        probs = torch.nn.functional.softmax(predictions_list, dim=1)
                
    # probs = torch.nn.functional.softmax(predictions_list, dim=1)

    results = metrics.classification_report(
        labels_list, torch.argmax(probs, dim=1), output_dict=True  # predictions_list
    )
    cm = metrics.confusion_matrix(
        labels_list, torch.argmax(probs, dim=1)
    )


    auc_score = metrics.roc_auc_score(labels_list, probs[:, 1])
    auprc_score = metrics.average_precision_score(labels_list, probs[:, 1])
    accuracy_score = metrics.accuracy_score(labels_list, np.argmax(probs, axis=1))

    print(results)
    print(cm)
    print(f"Test Loss: {loss}")
    print(f"Accuracy: {accuracy_score}")
    print(f"AUPRC: {auprc_score}")
    print(f"AUROC: {auc_score}")

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
