import torch.nn as nn

class FCNetwork(nn.Module):
	def __init__(self,n_inputs,n_outputs,hidden_layers=[],**kwargs):
		super().__init__()
		net = []
		layers = [n_inputs] + hidden_layers + [n_outputs]
		layer = n_inputs

		for next_layer in hidden_layers:
			net += [nn.Linear(layer,next_layer),nn.ReLU()]
			layer = next_layer
		net += [nn.Linear(layer,n_outputs)]

		self.pipe = nn.Sequential(*net)

	def forward(self,x,*args):
		return self.pipe(x), None