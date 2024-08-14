import torch.nn as nn
import numpy as np
import torch
import torch.utils
from NNs.autoencoder import *
from lib.train import *
from lib.helper import *
import random
from itertools import product
from lib.data_loading import *
from config.vae import VAE_config
from torch.utils.tensorboard import SummaryWriter
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##Hyperparameters:
part_class = VAE_config.part_class
batch_size= VAE_config.batch_size
num_epochs = VAE_config.num_epochs
model_name = VAE_config.model_name   
parameters = dict(  
    latent_dim = [VAE_config.latent_dim],
    base = [VAE_config.base],
    lr = [VAE_config.lr]
)
##Data
data_train = VAE_config.data_train
data_test = VAE_config.data_test
mean = torch.mean(data_train)
##Scale data by mean
data_train = data_train/mean
data_test = data_test/mean
#Use subset of data for training
data_train = data_train[:,:500]
data_test = data_test[:,:500]

time = torch.linspace(1e-07,1e-04,1000)
##combine timesteps and config dim for training of the ae model
data = torch.reshape(data_train,(data_train.shape[0]*data_train.shape[1],1,50,100))
##Normalize data


time_train = torch.repeat_interleave(time, data_train.shape[0], dim=0) #repeat for each config --> 51 in training


train_size = int(0.8 * len(data))
val_size = len(data) - train_size
data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])
##Corresponds almost to 80/20 split and 
#ensures that the configs are not split within

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=val_size, shuffle=True)



time_test = torch.repeat_interleave(time, data_test.shape[0], dim=0) #repeat for each config --> 13 in training

data_test = (np.reshape(data_test,(data_test.shape[0]*data_test.shape[1],1,50,100)))
#data_test  =torch.utils.data.TensorDataset(data_test)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)


##Create model


criterion = nn.MSELoss(reduction='sum')


## hyper parameter loop

for run_id, (latent_dim, base, lr) in enumerate(product(*parameters.values())):
    encoder = pro_encoder2d(part_class,base, latent_dim)
    decoder = pro_decoder2d(part_class,base, latent_dim)
    model = VAE(encoder, decoder, latent_dim=latent_dim, mode="train")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ##number of parameters in the model
    n_params = sum(p.numel() for p in model.parameters())
    print("Number of trainable parameters", n_params)
    comment = f' latent_dim={latent_dim} base={base} lr={lr}'
    tb = SummaryWriter(comment=comment) # tensorboard writer
    model.train()
    train_vae(model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs, model_name=model_name)
    model.cpu() # move the model to the cpu
    #torch.save(best_model, 'model_vae_lin.pth')
    model.eval()
    model.load_state_dict(torch.load(model_name, map_location=device))
    test_loss = 0
    for x_test in test_loader:
            # Get the inputs; data is a list of [inputs, labels]
            #data = data.unsqueeze(1)
            #x_test = data.to(x_test)
            decoded, _,_,_ = model(x_test)
            test_loss += criterion(decoded, x_test)#+ (torch.sum(y_train!=y_pred))
    
    test_loss = test_loss/len(test_loader.dataset)
    print("Test loss", test_loss.item())
    #tb.add_hparams(
    #{"latent_dim": latent_dim, "base": base, "lr": lr},{"Train loss": train_loss, "Train KLD": train_KLD,  "Test loss": test_loss.item()}
    #)
tb.close()



