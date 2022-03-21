import torch
import torch.nn as nn


class EncoderNN(nn.Module):
	def __init__(self,input_size,output_size,num_layers=1,
				pre_processing_layers=[],post_processing_layers=[],**kwargs):
		super().__init__()

		if pre_processing_layers == []:
			self.pre_net = nn.Sequential()
		else:
			pre_processing_network = self.build_network(input_size,pre_processing_layers[-1],pre_processing_layers[:-1])
			self.pre_net = nn.Sequential(*pre_processing_network,nn.ReLU())
			input_size = pre_processing_layers[-1]

		if post_processing_layers == []:
			self.post_net = nn.Sequential()
		else:
			post_processing_network = self.build_network(post_processing_layers[0],output_size,post_processing_layers[1:])
			self.post_net = nn.Sequential(nn.ReLU(),*post_processing_network)
			output_size = post_processing_layers[0]

		self.rnn = nn.LSTM(input_size,output_size,num_layers=num_layers)

	def forward(self,data):

		if not isinstance(data,list):
			data = [data]

		output = []
		for x in data:
			out = self.pre_net(x)
			self.rnn.flatten_parameters()
			out, _ = self.rnn(out)
			out = self.post_net(out[-1,:,:])
			output.append(out)
		output = torch.stack(output)

		return output.squeeze(1)


	def build_network(self,input_size,output_size,layers):
		net = []
		layer_size = input_size

		for next_layer_size in layers:
			net += [nn.Linear(layer_size,next_layer_size),nn.ReLU()]
			layer_size = next_layer_size

		net += [nn.Linear(layer_size,output_size)]
		return net

class DecoderNN(nn.Module):

	def __init__(self,input_size,decoder_hidden_layers=[],**kwargs):
		super(DecoderNN,self).__init__()
		net = []
		layer = input_size

		for next_layer in decoder_hidden_layers:
			net += [nn.Linear(layer,next_layer),nn.ReLU()]
			layer = next_layer
		net += [nn.Linear(layer,1)]

		self.pipe = nn.Sequential(*net)
		self.sigmoid = nn.Sigmoid()

	def forward(self,data,x):
		data = torch.cat([data,x],dim=1)
		return self.pipe(data)

	def predict(self,data,x):
		data = torch.cat([data,x],dim=1)
		return torch.round(self.sigmoid(self.pipe(data)))

if __name__ == '__main__':

	input_size = 2
	pre_layers = [5,5]
	post_layers = [5,10]
	output_size = 5

	decoder_hidden_layers = [10]

	sequence_lengths = [1,2,3]

	data = []
	for sequence_length in sequence_lengths:
		data.append(torch.rand(sequence_length,1,input_size))

	encoder = EncoderNN(input_size,output_size,pre_processing_layers=pre_layers,post_processing_layers=post_layers)
	output = encoder(data)
	# print(encoder)
	print(output.shape)

	decoder = DecoderNN(output_size,hidden_layers=decoder_hidden_layers)
	print('\n',decoder(output))
	print(decoder.predict(output))