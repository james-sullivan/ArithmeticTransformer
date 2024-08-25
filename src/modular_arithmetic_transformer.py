import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import CharDelimiterSplit

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ModularArithmeticDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        equation = item['input']
        result = item['output']

        # Tokenize equation
        encoded = self.tokenizer.encode(equation)
        equation_tokens = encoded.ids

        return torch.tensor(equation_tokens, dtype=torch.long), torch.tensor(int(result), dtype=torch.long)


class ArithmeticTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, max_result):
        super(ArithmeticTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(d_model, max_result)

    def forward(self, src):
        src = src.permute(1, 0)  # Change to (seq_len, batch_size)

        # Create positional encodings
        positions = torch.arange(0, src.size(0)).unsqueeze(1).expand(src.size()).to(src.device)

        # Combine token embeddings and positional encodings
        x = self.embedding(src) + self.pos_encoder(positions)

        # Pass through the transformer
        output = self.transformer_encoder(x)

        # Use the output of the last token for classification
        output = output[-1, :, :]

        # Project to the number of possible results
        output = self.fc(output)

        return output

if __name__ == "__main__":
    # Get paths to the test and train data
    data_path = os.path.join("..", "data")
    train_data_path = os.path.join(data_path, 'modular_arithmetic_train.csv')
    test_data_path = os.path.join(data_path, 'modular_arithmetic_test.csv')

    # Load and preprocess the data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Initialize and train the tokenizer
    tokenizer = Tokenizer(Unigram())
    trainer = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = CharDelimiterSplit("")  # Split each character

    # Prepare the data for tokenizer training
    def iterator():
        for equation in train_data['input']:
            yield equation

    tokenizer.train_from_iterator(iterator(), trainer)

    # Create datasets and dataloaders
    train_dataset = ModularArithmeticDataset(train_data, tokenizer)
    test_dataset = ModularArithmeticDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model parameters
    vocab_size = tokenizer.get_vocab_size()
    d_model = 64
    nhead = 1
    num_layers = 3
    dim_feedforward = 256
    max_seq_length = 20  # Adjust based on your longest equation
    max_result = train_data['output'].max() + 1  # +1 because we start counting from 0

    # Initialize the model
    model = ArithmeticTransformer(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, max_result)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for equation, result in train_loader:
            equation, result = equation.to(device), result.to(device)
            optimizer.zero_grad()
            output = model(equation)
            loss = criterion(output, result)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for equation, result in test_loader:
                equation, result = equation.to(device), result.to(device)
                output = model(equation)
                _, predicted = torch.max(output, 1)
                total += result.size(0)
                correct += (predicted == result).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    # Test the model
    model.eval()
    with torch.no_grad():
        for equation, result in test_loader:
            equation, result = equation.to(device), result.to(device)
            output = model(equation)
            _, predicted = torch.max(output, 1)
            for i in range(len(equation)):
                eq_str = tokenizer.decode(equation[i].tolist())
                print(f"Equation: {eq_str}, Predicted: {predicted[i].item()}, Actual: {result[i].item()}")