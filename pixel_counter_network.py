import torch, os, cv2
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10000)  # 1 input -> 1000 hidden neurons
        self.relu = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(10000, 1)  # 1000 hidden -> 1 output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

coreDir = os.path.expanduser("~/Downloads/MSR/CustomI2GROW_Dataset/")
plants = sorted(os.listdir('/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/RGBDImages/'))
trueDir = "/Biomass_Info_Ground_Truth.csv"
trueData = pd.read_csv(coreDir + trueDir)
num_pixels = []
for p in tqdm(plants):
    rgbd = np.load('/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/RGBDImages/'+p)
    rgb = rgbd[:,:,:3].astype(np.uint8)
    rgb_ = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35,35,35])
    upper_green = np.array([85,255,255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5,5), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel_dilation = np.ones((5,5), np.uint8)
    mask_opened = cv2.dilate(mask_opened, kernel_dilation, iterations=1)
    true_mass = float(trueData[trueData['Data ID'] == p.replace('.npy','')]['Fresh Biomass'].iloc[0])
    if true_mass <= 30:
        num_pixels.append((np.sum(mask_opened)/255, int(true_mass)))
    

# Create model
model = SimpleNN()

# Define loss function (MSE for regression)
criterion = nn.MSELoss()

# Define optimizer (Adam works well)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate some synthetic training data (for example)
# x_train = torch.randn(100, 1) * 10  # 100 random inputs (scaled)
# y_train = 2 * x_train + 3           # Example function: y = 2x + 3
random.seed(5)
random.shuffle(num_pixels)
x = [int(p[0]) for p in num_pixels]
y = [int(p[1]) for p in num_pixels]

x_train = torch.tensor(x[:int(.8*len(x))], dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y[:int(.8*len(y))], dtype=torch.float32).unsqueeze(1)
x_valid = torch.tensor(x[int(.8*len(x)):int(.9*len(x))], dtype=torch.float32).unsqueeze(1)
y_valid = torch.tensor(y[int(.8*len(y)):int(.9*len(y))], dtype=torch.float32).unsqueeze(1)
x_test = torch.tensor(x[int(.9*len(x)):], dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y[int(.9*len(y)):], dtype=torch.float32).unsqueeze(1)

losses = [[],[]]
# Training loop
epochs = 1000  # Number of training iterations
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Reset gradients

    outputs = model(x_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss

    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    model.eval()
    with torch.no_grad():
        val_outputs = model(x_valid)
        val_loss = criterion(val_outputs, y_valid)

    if (epoch + 1) % 100 == 0:  # Print every 100 epochs
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {np.sqrt(val_loss):.4f}")

    losses[0].append(np.sqrt(loss.item()))
    losses[1].append(np.sqrt(val_loss))

model.eval()
with torch.no_grad():
    test_outputs = model(x_test)
    test_loss = criterion(test_outputs, y_test)

print(f"Testing, Loss: {np.sqrt(test_loss):.4f}")
print(r2_score(test_outputs, y_test))

print("Training complete!")

plt.figure(0)
plt.plot(np.arange(0,epochs, 1), losses[0], label="Training Loss")
plt.plot(np.arange(0,epochs, 1), losses[1], label="Validation Loss")
plt.yscale('log')
plt.title("Pixel Counting Testing")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

plt.figure(1)
plt.scatter(test_outputs, y_test)
plt.plot(np.arange(0,torch.max(test_outputs)), np.arange(0,torch.max(test_outputs)), 'k-')
plt.title("Actual v Predicted")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.xlim([0,torch.max(test_outputs)])
plt.ylim([0,torch.max(test_outputs)])
plt.show()