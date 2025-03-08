import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerModel, LongformerTokenizer
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim import AdamW
from tqdm import tqdm
import os

# Load the dataset
df = pd.read_csv(r'/dataset.csv', encoding='latin-1')

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Preprocess the text: tokenize and pad/truncate
max_length = 64
df['body'] = df['body'].apply(lambda x: tokenizer.encode_plus(x, truncation=True, padding='max_length', max_length=max_length))

# Drop 'author' column
df.drop(['author'], axis=1, inplace=True)

# Split into train and test first
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

# Split df_train into train and validation
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

# Define a PyTorch dataset
class PersonalityDataset(Dataset):
    def __init__(self, tweets, targets):
        self.tweets = tweets
        self.targets = targets

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tweets[idx]['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(self.tweets[idx]['attention_mask'], dtype=torch.long)
        targets = torch.tensor(self.targets[idx], dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': targets
        }

# Define the model
class RoBERTaForPersonalityTraits(torch.nn.Module):
    def __init__(self):
        super(RoBERTaForPersonalityTraits, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        output = self.linear(output)
        return output

# Prepare data loaders
batch_size = 16
train_dataset = PersonalityDataset(df_train['body'].tolist(), df_train[['ext', 'neu', 'agr', 'con', 'ope']].values)
val_dataset = PersonalityDataset(df_val['body'].tolist(), df_val[['ext', 'neu', 'agr', 'con', 'ope']].values)
test_dataset = PersonalityDataset(df_test['body'].tolist(), df_test[['ext', 'neu', 'agr', 'con', 'ope']].values)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

# Instantiate the model and define optimizer and loss function
model = RoBERTaForPersonalityTraits()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.MSELoss()

# Training loop
epochs = 10
patience = 3
best_val_loss = float('inf')
patience_counter = 0
accumulation_steps = 16

# Output log file
log_file = r'/roberta_base_result.txt'

# Open the log file for writing
with open(log_file, 'w') as f:
    f.write("Training started\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        step = 0

        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")
        for i, batch in enumerate(train_progress_bar):
            optimizer.zero_grad() if i % accumulation_steps == 0 else None

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)

            # Backward pass
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or i + 1 == len(train_loader):  # Update every accumulation step
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            step += 1

            # Save model every 5000 steps
            if step % 5000 == 0:
                torch.save(model.state_dict(), f'roberta_base/model_step_{step}.pth')

            train_progress_bar.set_postfix({'loss': total_loss / (i + 1)})

        avg_train_loss = total_loss / len(train_loader)
        f.write(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss}\n")

        # Validation loop
        model.eval()
        val_total_loss = 0
        all_predictions = [[] for _ in range(5)]
        all_targets = [[] for _ in range(5)]

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation")
        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, targets)

                val_total_loss += loss.item()

                for i in range(5):
                    all_predictions[i].extend(outputs[:, i].cpu().numpy())
                    all_targets[i].extend(targets[:, i].cpu().numpy())

                val_progress_bar.set_postfix({'val_loss': val_total_loss / len(val_loader)})

        avg_val_loss = val_total_loss / len(val_loader)
        f.write(f"Epoch {epoch + 1}: Validation Loss: {avg_val_loss}\n")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), r'/roberta_base_best_model.pth')  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                f.write("Early stopping triggered.\n")
                break

    # Save final model
    torch.save(model.state_dict(), r'/roberta_base_final_model.pth')
    f.write("Final model saved.\n")

    # Evaluation on test set
    model.eval()
    test_total_loss = 0
    all_predictions = [[] for _ in range(5)]
    all_targets = [[] for _ in range(5)]

    test_progress_bar = tqdm(test_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in test_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)

            test_total_loss += loss.item()

            for i in range(5):
                all_predictions[i].extend(outputs[:, i].cpu().numpy())
                all_targets[i].extend(targets[:, i].cpu().numpy())

            test_progress_bar.set_postfix({'test_loss': test_total_loss / len(test_loader)})

    trait_names = ["Ext", "Neu", "Agre", "Con", "Ope"]
    for i in range(5):
        predictions = all_predictions[i]
        targets = all_targets[i]

        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        f.write(f"For {trait_names[i]}:\n")
        f.write(f"MSE: {mse}\nRMSE: {rmse}\nMAE: {mae}\nR^2: {r2}\n\n")

    df_predictions = pd.DataFrame(np.transpose(all_predictions), columns=trait_names)
    correlation_matrix_predictions = df_predictions.corr()
    f.write("Correlation matrix for predicted values:\n")
    f.write(str(correlation_matrix_predictions))

    correlation_matrix = df[['ext', 'neu', 'agr', 'con', 'ope']].corr()
    f.write("\nCorrelation matrix for actual values:\n")
    f.write(str(correlation_matrix))
