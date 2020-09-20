from torch import nn

class BaseModule(nn.Module):
	def __getattr__(self, key):
		if key in dir(self):
			return super().__getattr__(key)
		for name, value in self.named_children():
			if name.split("_")[0] == key:
				return value
		return super().__getattr__(key)
