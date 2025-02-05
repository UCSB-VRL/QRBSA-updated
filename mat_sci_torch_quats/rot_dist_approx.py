
import torch
from mat_sci_torch_quats.quats import hadamard_prod, conjugate, outer_prod, fz_reduce

class GradientApproxInvCos():

	# Q: multi-dimensional tensor of q
	# q = [scalar,v1,v2,v3]

	# t: gradient of inverse-cosine blows up near 1 and -1


	def __init__(self, beta=0.1, mode=True):

		## ex: if beta = 0.1, we will use a first order linear approximation of inverse-cosine at [0.9, 1] and [-1, -0.9]

		# import pdb; pdb.set_trace()

		self.mode = mode

		if self.mode:

			self.beta = beta
			self.mode = mode

			self.t1= 1- beta
			self.t2 = -1 + beta
			t1 = torch.tensor(1 - beta, dtype=torch.float32, requires_grad=True)
			t2 = torch.tensor(-1 + beta, dtype=torch.float32, requires_grad=True)
			
			y1 = 2*torch.arccos(t1)
			y2 = 2*torch.arccos(t2)

			y1.backward()
			y2.backward()

			self.m1 = float(t1.grad)
			self.m2 = float(t1.grad)

			self.b1 = float(y1) - self.m1 * self.t1
			self.b2 = float(y2) - self.m2 * self.t2

	def __call__(self, input):

		if self.mode:

			y_rot = 2*torch.arccos(input)

			y_lin_approx1 = self.m1 * input + self.b1
			y_lin_approx2 = self.m2 * input + self.b2

			y_out = y_rot * (torch.logical_and(input >= -1 + self.beta, input <= 1 - self.beta)).float()  \
					+ y_lin_approx1 * (input < -1 + self.beta).float() \
					+ y_lin_approx2 * (input > 1 - self.beta).float()

		else:
			with torch.no_grad():
				# import pdb; pdb.set_trace()
				y_out = 2 * torch.arccos(input)

		return y_out
        

# Callable object design pattern.
class MinimumAngleTransformation(torch.nn.Module):

	def __init__(self, mode):

		# import pdb; pdb.set_trace()
		self.mode = mode
		super(MinimumAngleTransformation, self).__init__()

		self.dist = GradientApproxInvCos(mode = self.mode)
		

	def forward(self, qSR, qHR, syms):

		# qSR = fz_reduce(qSR, syms) ## FZ reduction now also happens during training.

		T = hadamard_prod(conjugate(qHR), qSR)
		T_syms = outer_prod(T, syms)
		T_syms_scalar = T_syms[...,0]
		T_scalar_max, _ =  torch.max(T_syms, dim=-2)

		return self.dist(T_scalar_max)