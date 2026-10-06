import numpy as np
import torch
from math import cos, sin, tanh

class RBFNet:
    def __init__(self, 
                 popsize, 
                 num_basis,
                 num_output,
                 motor_encode='direct',
                 controlInterval=0.02 , # policy control frequency 
                 ):
        """
        Support only *DIRECT ENCODING*
        """
        self.architecture = [num_basis, num_output] # input was fix at 2 dimensional (CPG 2 phase)
        self.popsize = popsize

        # Initialize CPG
        self.phi = 0.084 # 0.084 for fast , 0.038 for slow
        self.controlInterval = controlInterval # 50Hz
        self.O = torch.Tensor([[0.0, 0.18]]).expand(popsize, 2).cuda()
        self.t, self.x, self.y, self.period = self.pre_compute_cpg(self.phi , self.controlInterval)

        # Rbf network
        self.num_basis = num_basis
        self.num_output = num_output
        self.variance = 25.0
        self.phase = 0

        # Pre calculated rbf layers output 
        # -- placing center, variance of RBF kernel :3
        self.ci, self.cx, self.cy, self.rx, self.ry, self.KENNE = self.pre_rbf_centers(
            self.period, self.num_basis, self.x, self.y, self.variance)
        self.KENNE = self.KENNE.cuda()

        # initilize motor encoding type (weights, CPGs' phase)
        self.motor_encode = motor_encode # 'direct', 'indirect'
        if self.motor_encode == 'direct':
            self.weights = torch.Tensor(popsize, num_basis, num_output).uniform_(-0.04, 0.04).cuda()
            # Initilize phase of each CPG
            phase_2 = int(self.period//2)
            self.phase = torch.Tensor([0, phase_2])

    def forward(self, pre):
        # print('pre: ', pre)
        with torch.no_grad():
            # Direct encoding ##################################
            p1 = self.KENNE[int(self.phase[0])]
            # p2 = self.KENNE[int(self.phase[1])]
 
            out_p1 = torch.tanh(torch.matmul(p1, self.weights))
            # out_p2 = torch.tanh(torch.matmul(p2, self.weights))

            self.phase = self.phase + 1
            self.phase = torch.where(self.phase > self.period, 0, self.phase)
            ####################################################

        return out_p1.float().detach()
    
    def get_n_params_a_model(self):
        return len(self.get_a_model_params())

    def get_models_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()

    def get_a_model_params(self):
        p = torch.cat([ params.flatten() for params in self.weights[0]] )

        return p.cpu().flatten().numpy()
    
    def set_models_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params).float()
        # print('flat_params: ', flat_params.shape)

        popsize, basis, num_out = self.weights.shape
        self.weights = flat_params.reshape(popsize, basis, num_out).cuda()

    def set_a_model_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params).float()
        # print('flat_params: ', flat_params.shape)

        popsize, basis, num_out = self.weights.shape
        # print('flat_params.repeat(popsize, 1, 1): ', flat_params.repeat(popsize, 1, 1).shape)
        self.weights = flat_params.repeat(popsize, 1, 1).reshape(popsize, basis, num_out).cuda()
            
    
    def pre_compute_cpg(self, phi=0.06, controlInterval=0.02):
        # Run for one period
        alpha = 1.01         # SO(2) Alpha term
        w11   = alpha*cos(phi)
        w12   = alpha*sin(phi)
        w21   =-w12
        w22   = w11

        # w11 = 1.4
        # w22 = w11
        # w12 = 0.18+0.18
        # w21 = -w12
    
        x     = []
        y     = []
        t     = []
        t.append(0)
        x.append(-0.197)
        y.append(0.0)
        period = 0
        while y[period] >= y[0]:
            period = period+1
            t.append(period*controlInterval)
            x.append(tanh(w11*x[period-1]+w12*y[period-1]))
            y.append(tanh(w22*y[period-1]+w21*x[period-1]))
            
        while y[period] <= y[0]:
            period = period+1
            t.append(period*controlInterval)
            x.append(tanh(w11*x[period-1]+w12*y[period-1]))
            y.append(tanh(w22*y[period-1]+w21*x[period-1]))
        period = period
        return t, x, y, period
    
    def pre_rbf_centers(self, period, num_basis, x, y, var):
        KENNE  = [0]*num_basis  # Kernels
        ci = np.asarray(np.around(np.linspace(1, period, num_basis+1)), dtype=int)

        ci = ci[:-1]

        cx = [0] * (len(ci))
        cy = [0] * (len(ci))
        cxy = [0] * (len(ci))

        # Placing all the RBF's center
        for k in range(len(ci)):
            cx[k] = x[ci[k]]
            cy[k] = y[ci[k]]

        for i in range(num_basis):
            rx   = [q - cx[i] for q in x]
            ry   = [q - cy[i] for q in y]
            KENNE[i] = np.exp(-(np.power((rx),2) + np.power((ry),2))*var)
        # whole period, center of X , center of Y, diff X , diff Y, RBF function
        return ci, cx, cy, rx, ry, torch.from_numpy(np.array(KENNE).T).float()

# Example usage ####################################
# rbf_net = RBFNet(popsize=1,
#                  num_basis=10,
#                  num_output=18,
#                  robot='Ant',
#                  motor_encode='semi-indirect')

# pre = torch.randn(1, 2)  # Example dummy input
# import matplotlib.pyplot as plt

# outputs = []
# for i in range(200):
#     output = rbf_net.forward(pre)
#     outputs.append(output.cpu().numpy())

# outputs = np.concatenate(outputs, axis=0) if outputs[0].ndim == 1 else np.stack(outputs, axis=0)

# plt.figure(figsize=(12, 6))
# for j in range(outputs.shape[1]):
#     plt.plot(outputs[:, j], label=f'Joint {j}')
# plt.xlabel('Step')
# plt.ylabel('Output')
# plt.title('CPG-RBF Network Output Over Time')
# plt.legend()
# plt.show()

###################################################