import torch.nn as nn
import torch
from collections import OrderedDict


# class force_Smises_prediction_model(nn.Module):
#     def __init__(self, inputSize, outputSize):
#         super(force_Smises_prediction_model, self).__init__()

#         self.input_size = inputSize
#         # self.hidden_sizes = [32, 32, 32]
#         self.hidden_sizes = [1024,2048,4096]
#         self.output_size = outputSize
#         self.dropout_rate = 0.2

#         self.model = nn.Sequential(OrderedDict([
#             # ('BatchNormal0', nn.BatchNorm1d(self.input_size)),
#             ('fc0', nn.Linear(self.input_size, self.hidden_sizes[0])),
#             ('tanh0', nn.Tanh()),
#             # ('BatchNormal1', nn.BatchNorm1d(self.hidden_sizes[0])),
#             # ('dropout0', nn.Dropout(p=self.dropout_rate)),
#             ('fc1', nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
#             ('tanh1', nn.Tanh()),
#             # ('BatchNormal2', nn.BatchNorm1d(self.hidden_sizes[1])),
#             # ('dropout1', nn.Dropout(p=self.dropout_rate)),
#             ('fc2', nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])),
#             ('tanh2', nn.Tanh()),
#             ('output', nn.Linear(self.hidden_sizes[2], self.output_size))]))

#     def forward(self, x):
#         outputs = self.model(x)
#         return outputs


class tiny_model(nn.Module):
    def __init__(self,inputSize, outputSize):
        super(tiny_model, self).__init__() 
        self.layers = nn.Sequential(
            # nn.BatchNorm1d(inputSize),
            nn.Linear(inputSize, 4096),
        
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, outputSize)
        ) 
    def forward(self, x):  
        x = self.layers(x)
        return x

# class model_dynamic(nn.Module):
#     def __init__(self, inputSize, outputSize,Hiddens,Dropouts):
#         super(model_dynamic, self).__init__()

#         self.input_size = inputSize
#         # self.hidden_sizes = [32, 32, 32]
#         self.Hiddens = Hiddens
#         self.Dropouts = Dropouts
#         self.output_size = outputSize
#         self.layers = nn.ModuleList()
#         for index,i in enumerate(Hiddens):
#             if(index==0):
#                 self.layers.append(nn.Linear(self.input_size, self.Hiddens[index]))
#                 # self.layers.append(nn.BatchNorm1d(self.Hiddens[index]))
#                 self.layers.append(nn.ReLU())
            
#             elif(index==len(self.Hiddens)-1):
              
                
#                 # self.layers.append(nn.Dropout(self.Dropouts[index-1]))
#                 self.layers.append(nn.Linear(self.Hiddens[index-1], self.Hiddens[index]))
#                 # self.layers.append(nn.BatchNorm1d(self.Hiddens[index]))
#                 self.layers.append(nn.ReLU())
              
                
#                 # self.layers.append(nn.Dropout(self.Dropouts[index-1]))
#                 self.layers.append(nn.Linear(self.Hiddens[index], self.output_size))
                

#             else:
#                 # self.layers.append(nn.Dropout(self.Dropouts[index-1]))
#                 self.layers.append(nn.Linear(self.Hiddens[index-1], self.Hiddens[index]))
#                 # self.layers.append(nn.BatchNorm1d(self.Hiddens[index]))
#                 self.layers.append(nn.ReLU())
                
               
    

    # def forward(self, x):
    #     for layer in self.layers:
    #         x = layer(x)
    #     return x


