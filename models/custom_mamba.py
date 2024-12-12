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


torch.cuda.empty_cache()

#Structure the dataset as tensors
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
# It does batch preparation, padding of time-series data, since not all the data has the same length
def custom_collate_fn(batch):
    ts_values = [item[0] for item in batch]
    static_features = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])

    ts_values_padded = pad_sequence(ts_values, batch_first=True)
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)

    return ts_values_padded, static_features, labels, lengths


# This function acts as a tokenizer, because it discretizes the continuous variables into bins (quantization)
def quantize_tensor(tensor, num_bins):
   
    min_val, max_val = tensor.min(), tensor.max()

    #Adjust so all vals are in the positive range
    tensor_normalized = (tensor - min_val) / (max_val - min_val)

    #Create the bins (0,1) and discretize
    bins = torch.linspace(0, 1, steps=num_bins + 1, device=tensor.device)
    quantized = torch.bucketize(tensor_normalized, bins) - 1 
    #All the bins need to be in the appropiate range, we got an error if we did not do this.
    quantized = quantized.clamp(0, num_bins - 1) 

    #Reduce the dimensionality
    return quantized.argmax(-1)


# Training function, this just trains the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_bins = 128 
    for ts_values, static_features, labels, lengths in train_loader:
        ts_values, static_features, labels, lengths = (
            ts_values.to(device),
            static_features.to(device),
            labels.to(device),
            lengths.to(device),
        )

        ts_values_quantized = quantize_tensor(ts_values, num_bins)

        optimizer.zero_grad()

        # Forward pass

        #Attention mask for mamba. Telling the model to only focus on non-negative values.
        #Improvement point -> define the mask with the indicators in the data
        attention_mask = (ts_values_quantized > 0).float()  

        outputs = model(input_ids=ts_values_quantized, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]  # Use the last token's logits for classification
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluation Function -> same as training function but with some functions for evaluating the performance
def evaluate_with_metrics(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    num_bins = 50 
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
            logits = outputs.logits
            logits = logits[:, -1, :] 
            
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

# EarlyStopping Configuration -> using the earlystopping function from the main repo
early_stopping = EarlyStopping(patience=10, verbose=True, path="best_model.pt")



if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data = np.load('../P12data/split_1/split_1/train_physionet2012_1.npy', allow_pickle=True)
    validation_data = np.load('../P12data/split_1/split_1/validation_physionet2012_1.npy', allow_pickle=True)
    test_data = np.load('../P12data/split_1/split_1/test_physionet2012_1.npy', allow_pickle=True)

    # Create datasets and dataloaders
    train_dataset = EHRDataset(train_data)
    val_dataset = EHRDataset(validation_data)
    test_dataset = EHRDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=custom_collate_fn,num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=custom_collate_fn,num_workers=4, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True, shuffle=False)

    # Initialize MambaForCausalLM from scratch
    config = MambaConfig(
        vocab_size=5000, 
        n_positions=512, 
        hidden_size=256, 
        num_hidden_layers=8, 
        num_attention_heads=8, 
        intermediate_size=512,  
        hidden_dropout_prob=0.2, 
        attention_probs_dropout_prob=0.2,
    )

    model = MambaForCausalLM(config=config)
    model.to(device)
    
    #Adding the classification head to the model's layers
    model.lm_head = nn.Linear(config.hidden_size, 2).to(device)

    #Rebalancing the classes to help the model learn
    class_counts = Counter([item['labels'] for item in train_data])
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / count for count in class_counts.values()]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Loss, optimizer, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print("Starting training in epoch:", epoch)
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_acc, val_auroc, val_auprc, val_loss = evaluate_with_metrics(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation AUROC: {val_auroc:.4f}, Validation AUPRC: {val_auprc:.4f}")

        early_stopping(val_loss, model)
    
        #Stop if early stopping
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
        
        scheduler.step()

    #Load best model
    model.load_state_dict(torch.load("best_model_run1.pt"))
    # Test evaluation
    test_acc, test_auroc, test_auprc, test_loss = evaluate_with_metrics(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}, Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}, Test loss:  {test_loss:.4f}")