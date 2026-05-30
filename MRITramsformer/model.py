import torch
import torch.nn as nn
import torch.nn.functional as F

from MRITramsformer.utils.constants import INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, IN_CHANNELS, EMBEDDING, FEATURE_DIM, SPETIAL

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(INPUT_NUM_SAMPLES*INPUT_NUM_SHOTS*3, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, FEATURE_DIM)  # Second fully connected layer

    def forward(self, x):
        # Reshape the input
        current_num_frames, complex_dim, shots, samples_3 = x.size()
        flat_x = x.reshape(current_num_frames, complex_dim, shots * samples_3) # Shape: (1-8, 2, 8192)

        x = torch.relu(self.fc1(flat_x))
        x = torch.relu(self.fc2(x))

        x = x.permute(1,0,2) # Shape: (2, 1-8, 64)
        return x

class BasicTransformerTraj(nn.Module):
    def __init__(self):
        super(BasicTransformerTraj, self).__init__()
        self.encoder = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=FEATURE_DIM, kernel_size=3, stride=1, padding=1)
        # Adaptive pooling to reduce spatial dimensions to a fixed feature dimension
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch_size, FEATURE_DIM, 1, 1)
        
        self. transformer = nn.Transformer(d_model=FEATURE_DIM, nhead=4, num_encoder_layers=6)

        self.linear = nn.Linear(FEATURE_DIM, INPUT_NUM_SHOTS * INPUT_NUM_SAMPLES)

    def forward(self, x, num_dims=2):
        #reshaped_x = x.view(INPUT_NUM_FRAMES, 1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        x = x.permute(3, 0, 1, 2)
        encode_out = self.encoder(x)
        pooled_out = self.global_pool(encode_out)  # Shape: (batch_size, FEATURE_DIM, 1, 1)

        # Step 3: Reshape to (batch_size, 1, FEATURE_DIM)
        reshaped_encode_out = pooled_out.view(num_dims, 1, FEATURE_DIM)
       # reshaped_encode_out = encode_out.permute(2, 0, 3,
                                 #                1).contiguous()  # Shape: [INPUT_NUM_SHOTS, batch_size, INPUT_NUM_SAMPLES, OUT_CHANNELS]
        #reshaped_encode_out = reshaped_encode_out.view(batch_size,
                           #                            -1)  # Merge last two dims for d_model
        #reshaped_encode_out = reshaped_encode_out.unsqueeze(1)  # Add seq dim
        
        predictions = torch.zeros(num_dims, INPUT_NUM_FRAMES, FEATURE_DIM)
        # Autoregressive loop over frames
        current_input = reshaped_encode_out[:, :, :]  # Start with the first input

        for t in range(INPUT_NUM_FRAMES-1):
            
            output = self.transformer(current_input.permute(1, 0, 2), reshaped_encode_out.permute(1, 0, 2))
            output = output.permute(1,0,2)
            predictions[:, t, :] = reshaped_encode_out[:, 0, :]
           

            current_input = torch.cat((current_input[:, :, :], output[:, 0, :].unsqueeze(1)), dim=1)

            # Update current_input for next time step
            #current_input = output[:, 0, :, :].unsqueeze(1)
            
            
        current_input = self.linear(current_input)

        current_input = current_input.reshape(num_dims, INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        return current_input


class BasicTransformerShot(nn.Module):
    def __init__(self):
        super(BasicTransformerShot, self).__init__()

        self.encoder = nn.Conv2d(in_channels = IN_CHANNELS, out_channels = FEATURE_DIM, kernel_size = 3, stride = 1, padding = 1)
        self.transformer = nn.Transformer(d_model=FEATURE_DIM, nhead=4, num_encoder_layers=6)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(FEATURE_DIM, INPUT_NUM_SHOTS * INPUT_NUM_SAMPLES)


    def forward(self, x,  time_stamp=None):
        current_frames, shots, samples, dim = x.size()
        num_dims = 2
        x = x.permute(3, 0, 1, 2)
        output = torch.zeros(num_dims, 1, FEATURE_DIM)
        # reshaped_x = x.view(INPUT_NUM_FRAMES, 1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        encode_out = self.encoder(x)
        pooled_out = self.global_pool(encode_out)
        reshaped_encode_out = pooled_out.view(num_dims, 1, FEATURE_DIM)
        output = self.transformer(reshaped_encode_out.permute(1, 0, 2), output.permute(1, 0, 2))
        output = self.linear(output)
        output = output.permute(1, 0, 2).reshape(num_dims, 1, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
        return output

class AutoTrajPredictorAllFrames(nn.Module):
    def __init__(self, device, batch_size=6):
        super(AutoTrajPredictorAllFrames, self).__init__()
        self.transformer = nn.Transformer(d_model=EMBEDDING, nhead=4, num_encoder_layers=6)
        #self.linear = nn.Linear(FEATURE_DIM, INPUT_NUM_SHOTS * INPUT_NUM_SAMPLES)
        #self.mlp = MLP()
        self.device = device
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #.enc1 = self.conv_block(batch_size * 3, 256)
        self.enc2 = self.conv_block(batch_size, 513)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), )

    def forward(self, x):
        # x = frame_num, batch_size, feature_dim, 8*3
        frame_num, batch_size, embedding = x.size()
        x = x.permute(1, 0, 2)
        #x = x.permute(1, 0, 2, 3).reshape(batch_size * spetial, frame_num, feature_dim)
        output = torch.zeros(batch_size, 1, EMBEDDING).to(self.device)
        # x = self.mlp(x)
        output = self.transformer(x.permute(1, 0, 2), output.permute(1, 0, 2))
        # output = self.linear(output)
        output = output.reshape(1, batch_size, FEATURE_DIM, SPETIAL)
        #temp = self.pool(self.enc1(output))
        # output = self.enc2(output)
        # output = output.permute(0, 1, 2, 3).reshape(1 * 513 * 32, 1, 4)
        output = self.pool(self.enc2(output))
        # output = output.reshape(1, 513, 32, 2)
        # .reshape(NUM_DIMS, OUT_FRAME_NUM, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES))
        return output.permute(3, 0, 2, 1)

    
    
if __name__ == '__main__':
    model = BasicTransformerTraj()
    x = torch.randn(INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES)
    y = model(x)
    print(y.shape)
    