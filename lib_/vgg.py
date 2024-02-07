import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from data_loader import DataModule

# Define the VGG19 model
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = LoadVGG19()

    def forward(self, x):
        return self.vgg19(x)

# Load VGG19 model with modified architecture
def LoadVGG19():
    vgg19 = models.vgg19(pretrained=False)
    # Modify the first layer to accept 1 channel input
    vgg19.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    # Modify the fully connected layers
    num_features = vgg19.classifier[0].in_features
    vgg19.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 2),  # Change num_classes to the number of output classes
    )
    return vgg19

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct relative paths from the current directory
    content_dir = os.path.join(current_directory, 'lib_', 'dataset', 'content', "indian")
    style_dir = os.path.join(current_directory, 'lib_', 'dataset', 'style', "american")

    # Initialize PyTorch DataLoader
    dataloader = DataModule(current_directory, content_dir, style_dir, 8).train_dataloader()

    # Initialize the model
    model = VGG19()

    # Set device to CPU
    device = torch.device('cpu')
   # device = torch.device("cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch["content"].unsqueeze(1)
            targets = batch["style"].unsqueeze(1)
            #Stack the inputs and targets to one vector, with inputs label 1 and targets label 0
            stacked = torch.cat((inputs, targets), 0)
            one_hot_targets = torch.cat((torch.ones(inputs.shape[0]), torch.zeros(targets.shape[0])))
            #Convert one_hot_targets to float
            one_hot_targets = one_hot_targets.type(torch.LongTensor)
            #Create one hot targets, when the first 8 are 1 and the last 8 are 0
            stacked, targets = stacked.to(device), targets.to(device)

            # Forward pass
            outputs = model(stacked)
            loss = criterion(outputs, one_hot_targets)
            print('Current Loss:', loss.item())
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Print average loss at the end of each epoch
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#SAve the model weights as pth
    torch.save(model.state_dict(), os.path.join(current_directory, 'lib_', 'vgg19_new.pth'))
    print("Model weights saved to vgg19.pth")

exit(1)
