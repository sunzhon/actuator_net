import os
import pickle as pkl
from matplotlib import pyplot as plt
import time
import imageio
import numpy as np
from tqdm import tqdm
from glob import glob
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam

class ActuatorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['motor_states'])

    def __getitem__(self, idx):
        return {k: v[idx] for k,v in self.data.items()}

class Act(nn.Module):
  def __init__(self, act, slope=0.05):
    super(Act, self).__init__()
    self.act = act
    self.slope = slope
    self.shift = torch.log(torch.tensor(2.0)).item()

  def forward(self, input):
    if self.act == "relu":
      return F.relu(input)
    elif self.act == "leaky_relu":
      return F.leaky_relu(input)
    elif self.act == "sp":
      return F.softplus(input, beta=1.)
    elif self.act == "leaky_sp":
      return F.softplus(input, beta=1.) - self.slope * F.relu(-input)
    elif self.act == "elu":
      return F.elu(input, alpha=1.)
    elif self.act == "leaky_elu":
      return F.elu(input, alpha=1.) - self.slope * F.relu(-input)
    elif self.act == "ssp":
      return F.softplus(input, beta=1.) - self.shift
    elif self.act == "leaky_ssp":
      return (
          F.softplus(input, beta=1.) -
          self.slope * F.relu(-input) -
          self.shift
      )
    elif self.act == "tanh":
      return torch.tanh(input)
    elif self.act == "leaky_tanh":
      return torch.tanh(input) + self.slope * input
    elif self.act == "swish":
      return torch.sigmoid(input) * input
    elif self.act == "softsign":
        return F.softsign(input)
    else:
      raise RuntimeError(f"Undefined activation called {self.act}")

class Train():
    def __init__(self, 
            motor_num=12,
            data_sample_freq=100,
            datafile_dir=None,
            load_pretrained_model=False,
            device="cpu",
            **kwargs
            ):
        self.motor_num = motor_num
        self.data_sample_freq = data_sample_freq
        if(os.path.isfile(datafile_dir)):
            self.datafile_dir = os.path.dirname(datafile_dir)
        else:
            self.datafile_dir = datafile_dir
        self.load_pretrained_model = load_pretrained_model
        self.actuator_network_path = os.path.join(datafile_dir,"actuator.pt")
        self.device=device
        if("epochs" in kwargs.keys()):
            self.epochs = kwargs["epochs"]
        else:
            self.epochs = 1000

        if "display_motor_idx" in kwargs.keys():
            self.display_motor_idx = kwargs["display_motor_idx"]
        else:
            self.display_motor_idx = 0


    def build_mlp(self, in_dim, units, layers, out_dim,
              act='relu', layer_norm=False, act_final=False):
        mods = [nn.Linear(in_dim, units), Act(act)]
        for i in range(layers-1):
            mods += [nn.Linear(units, units), Act(act)]
        mods += [nn.Linear(units, out_dim)]
        if act_final:
            mods += [Act(act)]
        if layer_norm:
            mods += [nn.LayerNorm(out_dim)]
        return nn.Sequential(*mods)

    def train_actuator_network(self):
        """
        Train actuator model
        """
        print("model input dim: {} and output dim: {}".format( self.xs.shape, self.ys.shape))
    
        num_data = self.xs.shape[0]
        num_train = num_data // 5 * 4
        num_test = 500
        num_valid = num_data - num_train -num_test
    
        dataset = ActuatorDataset({"motor_states": self.xs, "motor_outputs": self.ys})
        train_set, val_set, self.test_set = torch.utils.data.random_split(dataset, [num_train, num_valid, num_test])
        train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        valid_loader = DataLoader(val_set, batch_size=128, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
    
        model = self.build_mlp(in_dim=self.xs.shape[1], units=100, layers=4, out_dim=self.ys.shape[1], act='softsign')
    
        lr = 8e-4
        opt = Adam(model.parameters(), lr=lr, eps=1e-8, weight_decay=0.0)
    
        model = model.to(self.device)
        for epoch in range(self.epochs):
            epoch_loss = 0
            ct = 0
            for batch in train_loader:
                data = batch['motor_states'].to(self.device)

                y_pred = model(data)
    
                opt.zero_grad()
    
                y_label = batch['motor_outputs'].to(self.device)
    
                tau_est_loss = ((y_pred - y_label) ** 2).mean()
                loss = tau_est_loss
    
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().cpu().numpy()
                ct += 1
            epoch_loss /= ct
    
            test_loss = 0
            mae = 0
            ct = 0
            if epoch % 1 == 0:
                with torch.no_grad():
                    for batch in valid_loader:
                        data = batch['motor_states'].to(self.device)
                        y_pred = model(data)
                        y_label = batch['motor_outputs'].to(self.device)
                        tau_est_loss = ((y_pred - y_label) ** 2).mean()
                        loss = tau_est_loss
                        test_mae = (y_pred - y_label).abs().mean()
    
                        test_loss += loss
                        mae += test_mae
                        ct += 1
                    test_loss /= ct
                    mae /= ct
                
                self.train_info = f'epoch: {epoch} | loss: {epoch_loss:.4f} | test loss: {test_loss:.4f} | mae: {mae:.4f}'
                print(self.train_info)
    
            model_scripted = torch.jit.script(model)  # Export to TorchScript
            model_scripted.save(self.actuator_network_path)  # Save
        return model



    def load_data(self):
        #1) load data
        data_path = os.path.join(self.datafile_dir, "motor_data.pkl")
        if(os.path.exists(data_path)):
            print("data file path:",data_path)
        else:
            print(data_path)
            warnings.warn("Data file path  not exists")
        with open(data_path, 'rb') as file:
            rawdata = pkl.load(file)
    
        self.xs = torch.tensor(rawdata["input_data"],dtype=torch.float).to(self.device)
        self.ys = torch.tensor(rawdata["output_data"],dtype=torch.float).to(self.device)
        self.timesteps = np.array(range(self.xs.shape[0])) / self.data_sample_freq


    
    def training_model(self):
        """
        Training model

        """
    
        # training model
        if self.load_pretrained_model:
            self.model = torch.jit.load(self.actuator_network_path).to(self.device)
        else:
            self.model = self.train_actuator_network().to(self.device)
    
    
    
    def eval_model(self):
        """
        Model evaluation

        """
    
        plot_length = 450
        prediction = []
        actual = []

        with torch.no_grad():
            for batch in self.test_loader:
                pred = self.model(batch["motor_states"]).detach()
                prediction.append(pred)
                actual.append(batch["motor_outputs"].detach())

        self.time = self.timesteps[:plot_length]
        self.actual = torch.tensor(actual)[:plot_length]
        self.prediction = torch.tensor(prediction)[:plot_length]

    

        """
        fig, axs = plt.subplots(1,1, figsize=(6, 4))
        axs = np.array(axs).flatten()
        #for i in range(self.motor_num):
        for i in range(2):
            axs[i].plot(timesteps, self.motor_torques[:plot_length, i], label="true torque")
            axs[i].plot(timesteps, tau_preds[:, i], linestyle='--', label="predicted torque")
            axs[i].set_xlabel("Time [s]")
            axs[i].set_ylabel("Torque [Nm]")
            axs[i].grid()
        plt.legend()
        plt.show()
        """
        #plt.savefig("esti.png")



if __name__=="__main__":
    kwargs={"epochs":1, "device":"cuda:0"}
    datafile_dir =  "./app/"
    training = Train(
            motor_num=12,
            data_sample_freq=100,
            datafile_dir = datafile_dir,
            load_pretrained_model = False,
            **kwargs
            )

    training.load_data()
    training.training_model()
    training.eval_model()

    import matplotlib.pyplot as plt
    plt.plot(training.time, training.actual, label="actual",color='r')
    plt.plot(training.time, training.prediction, label="prediction",color='g')
    plt.legend()
    plt.show()

