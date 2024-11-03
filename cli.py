from mortality_part_preprocessing import load_pad_separate
from mortality_classification import train_test, only_test
import os
import click
import torch
import random
import numpy as np
import json


@click.command()
@click.option(
    "--output_path",
    default="./ehr_classification_results/",
    help="Path to output folder",
)
@click.option("--pooling", default="max", help="pooling function")
@click.option("--epochs", default=300, help="model dropout rate")
@click.option("--dropout", default=0.4, help="model dropout rate")
@click.option("--attn_dropout", default=0.4, help="model attention dropout rate")
@click.option(
    "--model_type", default="vanilla", help="model_type"
)
@click.option("--heads", default=1, help="number of attention heads")
@click.option("--batch_size", default=64, help="batch size")
@click.option("--layers", default=1, help="number of attention layers")
@click.option("--dataset_id", default="physionet2012", help="filename id of dataset")
@click.option("--base_path", default="./P12data", help="Path to data folder")
@click.option(
    "--weight_decay", default=0.5, help="optimizer weight decay (regularization)"
)
@click.option("--lr", default=0.001, help="learning rate")
@click.option("--lr_patience", default=30, help="learning rate scheduler patience")
@click.option("--patience", default=10, help="patience for early stopping")
@click.option(
    "--use_mask",
    default=False,
    help="boolean, use mask for timepoints with no measurements",
)
@click.option(
    "--imputation", default=None, help="An optional method to use for imputation"
)  # currently broken
@click.option(
    "--use_static/--no_use_static", default=True, help="Use static features or not"
)
@click.option("--obs_strategy", default="both", type=click.Choice(["indicator_only", "obs_only", "both"]))
@click.option("--value_embed_size", default=8, help="size of sensor value embedding")
@click.option(
    "--early_stop_criteria",
    default="auroc",
    help="what to early stop on. Options are: auroc, auprc, auprc+auroc, or loss",
)
@click.option(
    "--positive_pass/--negative_pass",
    default=True,
    help="Whether to define an epoch by seeing 3x the positive samples or all the negative samples",
)
@click.option("--seft_n_phi_layers", default=3)
@click.option("--seft_phi_width", default=32)
@click.option("--seft_phi_dropout", default=0)
@click.option("--seft_n_psi_layers", default=2)
@click.option("--seft_psi_width", default=64)
@click.option("--seft_psi_latent_width", default=128)
@click.option("--seft_dot_prod_dim", default=128)
@click.option("--seft_latent_width", default=128)
@click.option("--seft_n_rho_layers", default=3)
@click.option("--seft_rho_width", default=32)
@click.option("--seft_rho_dropout", default=0)
@click.option("--seft_max_timescales", default=100)
@click.option("--seft_n_positional_dims", default=4)
@click.option("--ipnets_imputation_stepsize", default=0.25)
@click.option("--ipnets_reconst_fraction", default=0.25)
@click.option("--recurrent_dropout", default=0.3)
@click.option("--recurrent_n_units", default=100)
def core_function(
    output_path,
    base_path,
    model_type,
    epochs,
    dataset_id,
    batch_size,
    weight_decay,
    lr,
    lr_patience,
    patience,
    early_stop_criteria,
    positive_pass,
    imputation,
    **kwargs
):

    model_args = kwargs

    torch.manual_seed(0)  # 0
    random.seed(0)  # 0
    np.random.seed(0)  # 0

    accum_loss = []
    accum_accuracy = []
    accum_auprc = []
    accum_auroc = []

    for split_index in range(1, 6):

        base_path_new = f"{base_path}/split_{split_index}"
        train_pair, val_data, test_data = load_pad_separate(
            dataset_id, base_path_new, split_index, imputation
        )

        print(np.unique(train_pair.dataset_pos.label_array, return_counts=True))
        print(np.unique(train_pair.dataset_neg.label_array, return_counts=True))
        print(np.unique(val_data.label_array, return_counts=True))
        print(np.unique(test_data.label_array, return_counts=True))

        # make necessary folders
        parent_dir = "./results"
        directory = output_path
        model_path = os.path.join(parent_dir, directory)
        # if new model, make model folder
        if os.path.exists(model_path):
            pass
        else:
            try:
                os.mkdir(model_path)
            except OSError as err:
                print("OS error:", err)
        # make run folder
        base_run_path = os.path.join(model_path, f"split_{split_index}")
        run_path = base_run_path
        repeat_no = 0
        while os.path.exists(run_path):  # check for repeat
            repeat_no += 1
            run_path = base_run_path + f"_repeat_{repeat_no}"
        os.mkdir(run_path)

        # save model settings
        model_settings = {
            "model_type": model_type,
            "batch_size": batch_size,
            "batch_strategy": (
                "3x positive min" if positive_pass else "negative min"
            ),
            "epochs": epochs,
            "dataset": dataset_id,
            "weight_decay": weight_decay,
            "learning_rate": lr,
            "learning_rate_scheduler_patience": lr_patience,
            "patience": patience,
            "early_stop_criteria": early_stop_criteria,
            "base_path": base_path,
            "imputation": imputation,
            "dropout": model_args["dropout"],
            "pooling_fxn": model_args["pooling"],
            "heads": model_args["heads"],
            "layers": model_args["layers"],
            "use_timepoint_mask": model_args["use_mask"],
            "use_static_data": model_args["use_static"],
            "obs_strategy": model_args["obs_strategy"],
            "embedding_size": model_args["value_embed_size"],
        }
        if model_type == "seft":
            model_settings["seft_n_phi_layers"] = model_args["seft_n_phi_layers"]
            model_settings["seft_phi_width"] = model_args["seft_phi_width"]
            model_settings["seft_phi_dropout"] = model_args["seft_phi_dropout"]
            model_settings["seft_n_psi_layers"] = model_args["seft_n_psi_layers"]
            model_settings["seft_psi_width"] = model_args["seft_psi_width"]
            model_settings["seft_psi_latent_width"] = model_args["seft_psi_latent_width"]
            model_settings["seft_dot_prod_dim"] = model_args["seft_dot_prod_dim"]
            model_settings["seft_latent_width"] = model_args["seft_latent_width"]
            model_settings["seft_n_rho_layers"] = model_args["seft_n_rho_layers"]
            model_settings["seft_rho_width"] = model_args["seft_rho_width"]
            model_settings["seft_rho_dropout"] = model_args["seft_rho_dropout"]
        if model_type in ("grud", "ipnets"):
            model_settings["recurrent_dropout"] = model_args["recurrent_dropout"]
            model_settings["recurrent_n_units"] = model_args["recurrent_n_units"]
        if model_type == "ipnets":
            model_settings["ipnets_imputation_stepsize"] = model_args["ipnets_imputation_stepsize"]
            model_settings["ipnets_reconst_fraction"] = model_args["ipnets_reconst_fraction"]

        with open(f"{run_path}/model_settings.json", "w") as fp:
            json.dump(model_settings, fp)

        # run training
        loss, accuracy_score, auprc_score, auc_score = train_test(
            train_pair,
            val_data,
            test_data,
            output_path=run_path,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            lr=lr,
            lr_patience=lr_patience,
            patience=patience,
            early_stop_criteria=early_stop_criteria,
            positive_pass=positive_pass,
            model_args=model_args,
        )

        accum_loss.append(loss)
        accum_accuracy.append(accuracy_score)
        accum_auprc.append(auprc_score)
        accum_auroc.append(auc_score)


if __name__ == "__main__":
    core_function()  # modify as needed
