from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import numpy as np
from collections import Counter
from early_stopper import EarlyStopping
import random
import csv

torch.cuda.empty_cache()

# Dataset
class EHRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ts_values = torch.tensor(item['ts_values'], dtype=torch.float32)
        static = torch.tensor(item['static'], dtype=torch.float32)
        label = torch.tensor(item['labels'], dtype=torch.long)
        return ts_values, static, label

# Custom collate_fn for DataLoader
def custom_collate_fn(batch):
    ts_values = [item[0] for item in batch]
    static_features = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])

    ts_values_padded = pad_sequence(ts_values, batch_first=True)
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)

    return ts_values_padded, static_features, labels, lengths

# Quantization function
def quantize_tensor(tensor, num_bins):
    min_val, max_val = tensor.min(), tensor.max()
    tensor_normalized = (tensor - min_val) / (max_val - min_val)
    bins = torch.linspace(0, 1, steps=num_bins + 1, device=tensor.device)
    quantized = torch.bucketize(tensor_normalized, bins) - 1
    quantized = quantized.clamp(0, num_bins - 1)
    return quantized.argmax(-1)

# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_bins):
    model.train()
    total_loss = 0
    for ts_values, static_features, labels, lengths in train_loader:
        ts_values, static_features, labels, lengths = (
            ts_values.to(device),
            static_features.to(device),
            labels.to(device),
            lengths.to(device),
        )
        ts_values_quantized = quantize_tensor(ts_values, num_bins)
        optimizer.zero_grad()
        attention_mask = (ts_values_quantized > 0).float()
        outputs = model(input_ids=ts_values_quantized, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_with_metrics(model, loader, device, criterion, num_bins):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for ts_values, static_features, labels, lengths in loader:
            ts_values, static_features, labels, lengths = (
                ts_values.to(device),
                static_features.to(device),
                labels.to(device),
                lengths.to(device),
            )
            ts_values_quantized = quantize_tensor(ts_values, num_bins)
            outputs = model(input_ids=ts_values_quantized, attention_mask=(ts_values_quantized > 0))
            logits = outputs.logits[:, -1, :]
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    loss = total_loss / len(loader) if criterion is not None else None
    return acc, auroc, auprc, loss

# Logging function
def log_epoch_results(csv_path, trial, epoch, train_loss, val_loss, val_roc_auc_score):
    fieldnames = ["epoch", "train_loss", "val_loss", "val_roc_auc_score", "model", "split"]
    model_name = "custom_mamba"
    split_name = "split_1"
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_roc_auc_score": val_roc_auc_score,
        "model": model_name,
        "split": split_name,
    }
    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:  # File is new
            writer.writeheader()
        writer.writerow(row)

# Main loop
if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = np.load('../P12data/split_1/train_physionet2012_1.npy', allow_pickle=True)
    validation_data = np.load('../P12data/split_1/validation_physionet2012_1.npy', allow_pickle=True)
    test_data = np.load('../P12data/split_1/test_physionet2012_1.npy', allow_pickle=True)
    train_dataset = EHRDataset(train_data)
    val_dataset = EHRDataset(validation_data)
    test_dataset = EHRDataset(test_data)
    batch_size = 128 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True, shuffle=False)
    
    learning_rate_values = [1e-3, 5e-4]
    num_bins_values = [256, 512]
    num_trials = 4
    param_combinations = [{'learning_rate': random.choice(learning_rate_values), 'num_bins': random.choice(num_bins_values)} for _ in range(num_trials)]
    csv_path = "custom_mamba_last_results.csv"
    best_config = None
    best_metrics = {'AUROC': 0, 'AUPRC': 0}

    # Track best epoch results for the best trial
    best_epoch_results = []
    
    config = MambaConfig(vocab_size=5000, n_positions=512, hidden_size=512, num_hidden_layers=12, num_attention_heads=12, intermediate_size=512, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3)
    for trial, params in enumerate(param_combinations):
        print(f"Trial {trial + 1}/{num_trials} - Params: {params}")
        model = MambaForCausalLM(config=config).to(device)
        model.lm_head = nn.Linear(config.hidden_size, 2).to(device)
        class_counts = Counter([item['labels'] for item in train_data])
        total_samples = sum(class_counts.values())
        class_weights = torch.tensor([total_samples / count for count in class_counts.values()], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        early_stopping = EarlyStopping(patience=10, verbose=True, path=f"best_model_trial{trial + 1}.pt")
        
        trial_epoch_results = []

        for epoch in range(2):
            train_loss = train_model(model, train_loader, criterion, optimizer, device, params['num_bins'])
            val_acc, val_auroc, val_auprc, val_loss = evaluate_with_metrics(model, val_loader, device, criterion, params['num_bins'])
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}")

            trial_epoch_results.append((epoch + 1, train_loss, val_loss, val_auroc))

            early_stopping(val_auroc, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
            scheduler.step()


        model.load_state_dict(torch.load(f"best_model_trial{trial + 1}.pt"))
        val_acc, val_auroc, val_auprc, val_loss = evaluate_with_metrics(model, val_loader, device, criterion, params['num_bins'])

        if val_auroc > best_metrics['AUROC'] or (val_auroc == best_metrics['AUROC'] and val_auprc > best_metrics['AUPRC']):
            best_metrics = {'AUROC': val_auroc, 'AUPRC': val_auprc}
            best_config = params
            best_epoch_results = trial_epoch_results  # Save all epoch results for the best trial
            torch.save(model.state_dict(), "best_model_overall.pt")

    print(f"Best Hyperparameters: {best_config}")
    model.load_state_dict(torch.load("best_model_overall.pt"))
    test_acc, test_auroc, test_auprc, test_loss = evaluate_with_metrics(model, test_loader, device, criterion, params['num_bins'])
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}")

    # After all trials, log only the epoch results for the best trial
    if best_epoch_results:
        for epoch, train_loss, val_loss, val_roc_auc_score in best_epoch_results:
            log_epoch_results(csv_path, best_config, epoch, train_loss, val_loss, val_roc_auc_score)