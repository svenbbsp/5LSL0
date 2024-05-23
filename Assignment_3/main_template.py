# %% imports
# libraries
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader
import autoencoder_template

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = 'D://5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64
no_epochs = 4
learning_rate = 3e-4

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

print(f"Number of training batches: {len(train_loader)}"
        f"\nNumber of test batches: {len(test_loader)}")

# get some examples as the first 10 correspond to the digits 0-9
examples = enumerate(test_loader)
_, (x_clean_example, x_noisy_example, labels_example) = next(examples)

# show the examples in a plot
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2,10,i+11)
    plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("data_examples.png",dpi=300,bbox_inches='tight')
plt.show()

# %% create the autoencoder and optimizer
# create the autoencoder
AE = autoencoder_template.AE()

# create the optimizer
optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)

# Define the MSE loss function
loss_fn = nn.MSELoss()

# %% training loop

# Initialize a list to store the losses
minibatch_losses = []

# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    total_loss = 0
    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # fill in how to train your network using only the clean images
        # print size and shape of x_clean
        print(f"x_clean size = {x_clean.size()}")
        # Zero the gradients
        optimizer.zero_grad()

        # Pass the clean images through the autoencoder
        x_reconstructed, encoder_output = AE(x_clean)
        print(f"x_reconstructed size = {x_reconstructed.size()}")
        print(f"encoder_output size = {encoder_output.size()}")

        # Compute the loss
        loss = loss_fn(x_reconstructed, x_clean)

        # Backpropagate the loss
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the loss for this batch
        print(f"Batch {batch_idx}: Loss = {loss.item()}")

        # Store the loss for this minibatch
        minibatch_losses.append(loss.item())
        break

# Plot the loss for each minibatch over all epochs
plt.figure(figsize=(10, 5))
plt.plot(minibatch_losses)
plt.xlabel('Minibatch')
plt.ylabel('Loss')
plt.title('Loss for each minibatch over all epochs')
plt.show()


# # %% HINT
# # hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
# x_clean_train = train_loader.dataset.Clean_Images
# x_noisy_train = train_loader.dataset.Noisy_Images
# labels_train  = train_loader.dataset.Labels

# x_clean_test  = test_loader.dataset.Clean_Images
# x_noisy_test  = test_loader.dataset.Noisy_Images
# labels_test   = test_loader.dataset.Labels

# # use these 10 examples as representations for all digits
# x_clean_example = x_clean_test[0:10,:,:,:]
# x_noisy_example = x_noisy_test[0:10,:,:,:]
# labels_example = labels_test[0:10]