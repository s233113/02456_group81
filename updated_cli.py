import os
import json
import random
import numpy as np
import torch
import click
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from mortality_part_preprocessing import load_pad_separate
from mortality_classification import train_test
from transformers.models.mamba.modeling_mamba import MambaForCausalLM
from models.model_try import MambaFinetune
from models.transformer import TransformerModel
from models.seft import SEFTModel
from models.grud import GRUDModel
from models.ipnets import IPNetsModel


@click.command()
@click.option("--output_path", default="./ehr_classification_results/", help="Path to output folder")
@click.option("--pooling", default="max", help="Pooling function")
@click.option("--epochs", default=10, help="Number of training epochs")
@click.option("--dropout", default=0.4, help="Model dropout rate")
@click.option("--batch_size", default=32, help="Batch size")
@click.option("--dataset_id", default="physionet2012", help="Dataset ID")
@click.option("--base_path", default="./P12data", help="Path to data folder")
@click.option("--lr", default=5e-5, help="Learning rate")
@click.option("--num_labels", default=2, help="Number of labels for classification")
@click.option("--vocab_size", default=9, help="Vocabulary size")
@click.option("--model_type", default="mamba", help="Model type (mamba, transformer, seft, grud, ipnets)")
@click.option("--layers", default=1, help="Number of attention layers (transformer/seft)")
@click.option("--heads", default=1, help="Number of attention heads (transformer/seft)")
@click.option("--attn_dropout", default=0.4, help="Attention dropout rate (transformer/seft)")
@click.option("--patience", default=10, help="Patience for early stopping")
def core_function(
    output_path,
    pooling,
    epochs,
    dropout,
    batch_size,
    dataset_id,
    base_path,
    lr,
    num_labels,
    vocab_size,
    model_type,
    layers,
    heads,
    attn_dropout,
    patience,
):
    """
    CLI entry point for training the selected model.
    """
    # Set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Prepare dataset
    base_path_new = f"{base_path}/split_1"
    train_pair, val_data, test_data = load_pad_separate(dataset_id, base_path_new, split_index=1)

    # DataLoaders
    train_dataloader = DataLoader(train_pair, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model initialization
    if model_type == "mamba":
        pretrained_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        model = MambaFinetune(
            pretrained_model=pretrained_model,
            train_data=train_pair,
            problem_type="single_label_classification",
            num_labels=num_labels,
            vocab_size=vocab_size,
            learning_rate=lr,
            classifier_dropout=dropout,
        )
    elif model_type == "transformer":
        model = TransformerModel(
            input_dim=train_pair.data_array.shape[-1],
            num_classes=num_labels,
            num_heads=heads,
            num_layers=layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            pooling=pooling,
        )
    elif model_type == "seft":
        model = SEFTModel(
            input_dim=train_pair.data_array.shape[-1],
            num_classes=num_labels,
            layers=layers,
            heads=heads,
            pooling=pooling,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
    elif model_type == "grud":
        model = GRUDModel(
            input_dim=train_pair.data_array.shape[-1],
            num_classes=num_labels,
            dropout=dropout,
        )
    elif model_type == "ipnets":
        model = IPNetsModel(
            input_dim=train_pair.data_array.shape[-1],
            num_classes=num_labels,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Model type {model_type} is not supported.")

    # PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
    )

    # Create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save model settings
    model_settings = {
        "model_type": model_type,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "num_labels": num_labels,
        "vocab_size": vocab_size,
        "dropout": dropout,
        "attn_dropout": attn_dropout,
        "layers": layers,
        "heads": heads,
        "pooling": pooling,
        "patience": patience,
    }
    with open(f"{output_path}/model_settings.json", "w") as f:
        json.dump(model_settings, f)

    # Train the model
    print("Starting training...")
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save trained model
    model_save_path = os.path.join(output_path, f"{model_type}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate the model
    print("Starting evaluation...")
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    core_function()
