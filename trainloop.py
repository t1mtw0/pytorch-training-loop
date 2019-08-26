import time
from pbar import pbar

def train(model, trainloader, validloader, loss_function, optimizer, num_epochs):
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        print('Epoch ' + str(epoch))

        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

        model.train()

        train_loss = 0.0
        train_total = 0

        for i, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            targets = targets.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += targets.size(0)

            pbar(i * trainloader.batch_size,
                 (len(trainloader.dataset) // trainloader.batch_size) * trainloader.batch_size,
                 ['Loss'],
                 [round(loss.item(), 3)])

        print('\nTraining - Average Loss: {}'.format(
            round(train_loss / (train_total / trainloader.batch_size), 3)))

        model.eval()

        valid_loss = 0.0
        valid_total = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(validloader):
                inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                targets = targets.to('cuda' if torch.cuda.is_available() else 'cpu')

                outputs = model(inputs)
                loss = loss_function(outputs, targets)

                valid_loss += loss.item()
                valid_total += targets.size(0)

                pbar(i * validloader.batch_size,
                     (len(validloader.dataset) // validloader.batch_size) * validloader.batch_size,
                     ['Loss'],
                     [round(loss.item(), 3)])

        print('\nValidation - Average Loss: {}'.format(
            round(valid_loss / (valid_total / validloader.batch_size), 3)))

        print()

    total_time = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        total_time // 60, total_time % 60))

    return model
