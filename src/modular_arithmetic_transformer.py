import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ModularArithmeticDataset(Dataset):
    def __init__(self, data, encoder, max_seq_length, padding_char):
        self.data = data
        self.encoder = encoder
        self.max_seq_length = max_seq_length
        self.pad_token_id = self.encoder(padding_char)[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        equation = item['input']
        result = item['output']

        equation_tokens = self.encoder(equation)

        # Pad or truncate to max_seq_length
        if len(equation_tokens) < self.max_seq_length:
            equation_tokens += [self.pad_token_id] * (self.max_seq_length - len(equation_tokens))
        else:
            equation_tokens = equation_tokens[:self.max_seq_length]

        return torch.tensor(equation_tokens, dtype=torch.long), torch.tensor(int(result), dtype=torch.long)

class ArithmeticTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, max_result):
        super(ArithmeticTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(d_model, max_result)

    def forward(self, src):
        # src shape: [batch_size, seq_len]

        # Create a mask for padded elements
        mask = (src == 0).to(device)

        # Create positional encodings
        positions = torch.arange(0, src.size(1)).unsqueeze(0).expand(src.size()).to(device)

        # Combine token embeddings and positional encodings
        x = self.embedding(src) + self.pos_encoder(positions)

        # Pass through the transformer
        output = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Use the output of the last non-padded token for classification
        output = output[:, 0, :]  # Use the first token's output for classification

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

    # Get set of unique characters in training data input
    unique_characters = set("".join(train_data['input']))
    padding_char = "P"
    unique_characters.add(padding_char) # Add character for padding
    charToInt = {char: i for i, char in enumerate(unique_characters)}
    intToChar = {i: char for i, char in enumerate(unique_characters)}
    vocab_size = len(unique_characters)

    encoder = lambda string: [charToInt[char] for char in string]
    decoder = lambda arr: "".join([intToChar[i] for i in arr])

    print(f"vocab_size = {vocab_size}")
    print(f'encode("0123456789 +()=%") = {encoder("0123456789 +()=%")}')
    print(f'decode(encode("(44 + 83) % 10 =")) -> {decoder(encoder("(44 + 83) % 10 ="))}')

    # Determine max_length based on the longest equation in the dataset
    max_seq_length = train_data['input'].str.len().max()

    # Create datasets and dataloaders
    train_dataset = ModularArithmeticDataset(train_data, encoder)
    test_dataset = ModularArithmeticDataset(test_data, encoder)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model parameters
    d_model = 64
    nhead = 1
    num_layers = 3
    dim_feedforward = 256
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
'''
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
'''