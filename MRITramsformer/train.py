import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import BasicTransformerShot, BasicTransformerTraj
from MRITramsformer.utils.constants import INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, IN_CHANNELS, OUT_FRAME_NUM, FEATURE_DIM


# Dummy dataset generation (replace with real data)
def generate_dummy_data(num_samples):
    inputs = torch.rand(num_samples, IN_CHANNELS, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)  # Example shape
    targets = torch.rand(num_samples, INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)  # Example shape
    return inputs, targets

train_inputs, train_targets = generate_dummy_data(1000)
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=INPUT_NUM_FRAMES, shuffle=True)

def train(model, dataloader, criterion, optimizer, epochs=10, device="cpu"):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device if using GPU
            inputs, targets = inputs, targets

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}")

def train_autoregressive_trajectory(model, dataloader, criterion, optimizer, epochs=10, device="cpu"):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device if using GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize variables
            batch_size = inputs.size(0)
            predictions = torch.zeros_like(targets)  # To store predictions for the entire sequence

            # Initialize the autoregressive process with the first frame of the input
            optimizer.zero_grad()  # Clear gradients

            output = model(inputs, batch_size)

            # Compute loss
            loss = criterion(output, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}")

def train_autoregressive_shot(model, dataloader, criterion, optimizer, epochs=10, device="cpu", only_last_frame=False):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device if using GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize variables
            batch_size = inputs.size(0)
            predictions = torch.zeros_like(targets)  # To store predictions for the entire sequence

            # Initialize the autoregressive process with the first frame of the input
            current_input = inputs.clone()  # Start with the original input

            optimizer.zero_grad()  # Clear gradients

            # Autoregressive loop over frames
            for t in range(INPUT_NUM_FRAMES):
                # Forward pass
                if only_last_frame:
                    output = model(current_input, only_last_frame)
                else:
                    output = model(current_input, only_last_frame, t+1)

                # Store prediction
                predictions[:, t, :, :] = output[:, 0, :, :]

                # Update current_input for next time step
                if only_last_frame:
                    current_input = output[:, 0, :, :].unsqueeze(1)
                else:
                    current_input = torch.cat((current_input, output[:, :, :, :]), dim=0)

            # Compute loss
            loss = criterion(predictions, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}")


# Model, loss, and optimizer
if __name__ == '__main__':
    #model = BasicTransformerTraj()
    model = BasicTransformerShot()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_autoregressive_shot(model, train_loader, criterion, optimizer, epochs=10)
    #train_autoregressive_total(model, train_loader, criterion, optimizer, epochs=10)

    # Save the model
    torch.save(model.state_dict(), "basic_transformer_model.pth")
    print("Training complete. Model saved.")
