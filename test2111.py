from transformers.models.mamba.modeling_mamba import MambaForCausalLM
from models.model_try import MambaFinetune
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import numpy as np
from models.embeddings_try import preprocess_and_embed, get_vocab_size

outfile=open('myfile.txt', 'w')

def main(train_dataset, val_dataset):
    """
    Train the MambaFinetune model using preloaded datasets.
    
    :param train_dataset: Dataset object for training data
    :param val_dataset: Dataset object for validation data
    """
    # Hyperparameters
    batch_size = 32
    learning_rate = 5e-5
    num_epochs = 10

    # Load pretrained MambaForCausalLM model
    pretrained_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    # Initialize the fine-tuning model
    model = MambaFinetune(
        pretrained_model=pretrained_model,
        problem_type="single_label_classification",
        train_data=train_dataset,
        num_labels=2,
        num_tasks=6,
        learning_rate=learning_rate,
        classifier_dropout=0.1,
        multi_head=False,
        #change this!!!!!!!!
        vocab_size=9
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(train_dataloader)

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",  # Use "cpu" if no GPU is available
        devices=1,  # Number of GPUs to use, or None for automatic
        log_every_n_steps=10,
    )

    outfile.write("Starting training")
    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    # Replace with your dataset objects
    train_dataset = np.load('split_1/train_physionet2012_1.npy', allow_pickle=True)  # Your train dataset object
    val_dataset =  np.load('split_1/validation_physionet2012_1.npy', allow_pickle=True)   # Your validation dataset object
    
    main(train_dataset, val_dataset)
