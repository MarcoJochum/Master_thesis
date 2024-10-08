import torch.nn as nn
import torch
from torch.optim import lr_scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
# Define the autoencoder architecture
torch.manual_seed(42) 
def train_loop(train_x, model, criterion, optimizer, train_loader, num_epochs):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    # Loop over the dataset multiple times
    
    for epoch in range(num_epochs):
        # Initialize the running loss to 0
        running_loss = 0.0
        KLD = 0
        # Loop over the data loader
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            #train_loader is iterable, returns a batch of data at each iteration
            #data = data.unsqueeze(1)
            data = data.to(device)
            encoded, decoded, mu, log_var  = model(data)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = criterion(decoded, data) + 1* KLD

        
            # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*train_x.size(0)
        
        
        # Print the average loss for the epoche
        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()
        print('Epoch {}/{} Loss: {:.4f}  KLD: {:.4f}'.format(epoch+1,num_epochs, epoch_loss, 1*KLD))          


def train_loop_cae(train_x, model, criterion, optimizer, train_loader, num_epochs):
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.001)
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        # Initialize the running loss to 0
        running_loss = 0.0
        KLD = 0
        # Loop over the data loader
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            #data = data.unsqueeze(1)
            data = data.to(device)
            encoded, decoded  = model(data)
            y_train = torch.argmax(data, dim=1)
            y_pred = torch.argmax(decoded, dim=1)

            loss = criterion(decoded, data)+ (torch.sum(y_train!=y_pred))

        
            # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()#*data.size(0)
        
        #scheduler.step()
        # Print the average loss for the epoche
        epoch_loss = running_loss / len(train_loader.dataset)

        print('Epoch {}/{} Loss: {:.4f} '.format(epoch+1,num_epochs, epoch_loss, ))          

def train_loop_kmc(train_x, model, criterion, optimizer, train_loader, num_epochs):
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.001)
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        # Initialize the running loss to 0
        running_loss = 0.0
        KLD = 0
        # Loop over the data loader
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            
            x_train = data[0].to(device)
            y_train = data[1].to(device)
            #x_train = x_train.unsqueeze(1)
            
            encoded, decoded, mu, log_var  = model(x_train)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = criterion(decoded, y_train) + 1* KLD

        
            # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*train_x.size(0)
        
        #scheduler.step()
        # Print the average loss for the epoche
        epoch_loss = running_loss / len(train_loader.dataset)

        print('Epoch {}/{} Loss: {:.4f}  KLD: {:.4f}'.format(epoch+1,num_epochs, epoch_loss, 1*KLD))