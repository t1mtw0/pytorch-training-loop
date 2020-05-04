import torch
import time
from pbar import pbar

def train(model, train_loader, valid_loader, loss_function, optimizer, num_epochs):
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        print('Epoch ' + str(epoch))

        # Port model to the gpu
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Put model on train mode
        model.train()

        # Instantiate the train loss and total train samples
        train_loss = 0.0
        train_total = 0

        for i, (inputs, targets) in enumerate(train_loader):
            # Port data to gpu
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            targets = targets.to('cuda' if torch.cuda.is_available() else 'cpu')

            # Clear the weight gradients each batch of data
            optimizer.zero_grad()

            # Forward pass and loss
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            # Perform backward pass (calculating gradients of the weights) and use the optimizer to update weights
            loss.backward()
            optimizer.step()

            # Add loss and total samples
            train_loss += loss.item()
            train_total += targets.size(0)
            
            # Get number of batches in decimal form
            num_batches = len(train_loader.dataset) / train_loader.batch_size
            # If there is no excess samples, make the number of batches equal to the division, if not, add one to account for the excess
            if num_batches == int(num_batches):
                num_batches = int(num_batches)
            else:
                num_batches = int(num_batches) + 1
            # Check if number of items in the dataset is less than the batch size, if it is, change the batch size
            # Else, make the batch size the default
            if num_batches == 1:
                batch_size = len(train_loader.dataset)
            else:
                batch_size = train_loader.batch_size

            # Print the pbar and stats
            pbar((i+1) * batch_size,
                 num_batches * batch_size,
                 ['Loss'],
                 [round(loss.item(), 3)])

        # Print average loss of training
        print('\nTraining - Average Loss: {}'.format(
            round(train_loss / (train_total / train_loader.batch_size), 3)))

        # Put model in evaluation mode
        model.eval()

        # Instantiate validation loss and total number of validation samples
        valid_loss = 0.0
        valid_total = 0

        # Use no grad because of no weight updates
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(valid_loader):
                # Port data to gpu
                inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                targets = targets.to('cuda' if torch.cuda.is_available() else 'cpu')

                # Forward pass and loss calculation
                outputs = model(inputs)
                loss = loss_function(outputs, targets)

                # No backward pass and weight update
                
                # Add losses and total samples
                valid_loss += loss.item()
                valid_total += targets.size(0)
                
                # Get number of batches in decimal form
                num_batches = len(valid_loader.dataset) / valid_loader.batch_size
                # If there is no excess samples, make the number of batches equal to the division, if not, add one to account for the excess
                if num_batches == int(num_batches):
                    num_batches = int(num_batches)
                else:
                    num_batches = int(num_batches) + 1
                # Check if number of items in the dataset is less than the batch size, if it is, change the batch size
                # Else, make the batch size the default
                if num_batches == 1:
                    batch_size = len(valid_loader.dataset)
                else:
                    batch_size = valid_loader.batch_size
                    
                # Print the pbar and stats
                pbar((i+1) * batch_size,
                     num_batches * batch_size,
                     ['Loss'],
                     [round(loss.item(), 3)])

        # Print average loss of validation
        print('\nValidation - Average Loss: {}'.format(
            round(valid_loss / (valid_total / valid_loader.batch_size), 3)))

        print()

    # Print total time for training
    total_time = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        total_time // 60, total_time % 60))

    return model'
