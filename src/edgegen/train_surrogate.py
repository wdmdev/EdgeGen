import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import random
from pathlib import Path
import uuid
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from edgegen.utils import logging

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_data_csv_path", type=str, 
                        default='data/surrogate/56c3357c-466e-4c6b-a76d-0662293c3d4b_features.csv')

    args = parser.parse_args()

    model_id = uuid.uuid4()
    output_dir = Path(__file__).parent.parent.parent / 'output' / f'surrogate_models_{model_id}'
    logger = logging.get_logger(log_dir=output_dir, log_path_prefix='', name='train_surrogate')

    df = pd.read_csv(args.train_data_csv_path, sep=',', encoding='utf-8', decimal='.')

    # set Model Name as index
    df.set_index('Model Name', inplace=True)

    target_variables = ['Flash (kB)', 'RAM (kB)', 'Latency (ms)', 'Error (MAE)']    
    ys = df[target_variables]
    X = df.drop(columns=target_variables)

    seed = random.randint(0, 1000)
    logger.info(f'Random seed: {seed}')

    # Initialize a TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=str(output_dir))

    for target_variable in target_variables:
        print(f'Training surrogate model for {target_variable}')

        # Prepare data for training and testing.
        y = ys[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        input_size = X_train.shape[1]
        hidden_size = 64
        output_size = 1

        model = MLP(input_size, hidden_size, output_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        num_epochs = 10000
        for epoch in tqdm(range(num_epochs)):
            model.train()
            optimizer.zero_grad()

            inputs = torch.tensor(X_train, dtype=torch.float32)
            targets = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(-1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Log training loss to TensorBoard
            writer.add_scalar(f'{target_variable}/Training Loss', loss.item(), epoch)

        # Save the model
        model_save_path = output_dir / f'{target_variable}_surrogate_model.pt'
        torch.save(model.state_dict(), model_save_path)
        logger.info(f'Model saved at {model_save_path}')

        # Test model
        model.eval()
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        test_outputs = model(test_inputs)
        test_targets = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(-1)

        test_loss = criterion(test_outputs, test_targets)

        # Log test loss to TensorBoard
        writer.add_scalar(f'{target_variable}/Test Loss', test_loss.item(), 0)

        logger.info(f'{target_variable} Test Loss: {test_loss.item()}')

    # Log seed for reproducibility
    writer.add_text('Metadata', f'Random seed: {seed}')
    writer.close()

    logger.info(f'Surrogate models are saved in {output_dir}')