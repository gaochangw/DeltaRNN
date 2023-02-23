import torch
import torch.nn as nn
from networks.layers.deltalstm import DeltaLSTM
from networks.layers.deltagru import DeltaGRU

# Hyperparameters
rnn_layer = 'GRU'
input_size = 128
hidden_size = 512
num_layers = 3
thx = 0
thh = 0
dtype = torch.float32
torch.set_default_dtype(dtype)
device = "cpu"

# Generate Gaussian Random Inputs
batch_size = 1
seq_len = 100
mean = 0
std = 1
input = (torch.randn(seq_len, batch_size, input_size) - mean) * std
input = input.to(device)

# Compare
if rnn_layer == 'LSTM':
    # Instantiate Networks
    net_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    net_lstm = net_lstm.eval()
    net_lstm = net_lstm.to(device)
    net_deltalstm = DeltaLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, thx=thx, thh=thh)
    net_deltalstm = net_deltalstm.eval()
    net_deltalstm.load_state_dict(net_lstm.state_dict())
    net_deltalstm = net_deltalstm.to(device)
    # Forward Propagation
    out_deltalstm, (x_p_n_deltalstm, h_n_deltalstm, h_p_n_deltalstm, c_n_deltalstm, dm_n_deltalstm), _ = net_deltalstm(input)
    out_lstm, (h_n_lstm, c_n_lstm) = net_lstm(input)
    # Compute Error
    criterion = nn.MSELoss()
    rmse_out = torch.sqrt(criterion(out_lstm, out_deltalstm))
    rmse_h_n = torch.sqrt(criterion(h_n_lstm, h_n_deltalstm))
    rmse_c_n = torch.sqrt(criterion(c_n_lstm, c_n_deltalstm))
    print(rmse_out)
    print(rmse_h_n)
    print(rmse_c_n)
elif rnn_layer == 'GRU':
    # Instantiate Networks
    net_gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    net_gru = net_gru.eval()
    net_gru = net_gru.to(device)
    net_deltagru = DeltaGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, thx=thx, thh=thh)
    net_deltagru = net_deltagru.eval()
    net_deltagru.load_state_dict(net_gru.state_dict())
    net_deltagru = net_deltagru.to(device)
    # Forward Propagation
    out_deltagru, (x_p_n_deltagru, h_n_deltagru, h_p_n_deltagru, dm_nm_n_deltagru, dm_n_deltagru), _ = net_deltagru(input)
    out_gru, h_n_gru = net_gru(input)
    # Compute Error
    criterion = nn.MSELoss()
    rmse_out = torch.sqrt(criterion(out_gru, out_deltagru))
    rmse_h_n = torch.sqrt(criterion(h_n_gru, h_n_deltagru))
    print(rmse_out)
    print(rmse_h_n)
