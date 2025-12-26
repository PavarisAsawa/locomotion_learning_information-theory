import numpy as np
import torch
from math import cos, sin, tanh

def get_num_legjoints(robot):
    # Ant robot variation
    print("robot: ", robot)
    if robot == 'default':
        robot = 'Slalom'  # Default robot if not specified
    try:
        if robot == 'Slalom':
            num_legs = 4
            num_joints = 4 #6
            # """
            motor_mapping = torch.tensor([0,  4,  8,  12, 
                                          1,  5,  9,  13,  
                                          2,  6,  10, 14, 
                                          3,  7,  11, 15]
                                        )

        if (robot == 'pongbot' 
            or robot == 'anymal'
            or robot == 'pongbot_rbf'):
            num_legs = 4
            num_joints = 3 #6
            # """
            motor_mapping = torch.tensor([0,  6,  3, 9, 
                                          1,  7,  4, 10,  
                                          2,  8,  5, 11]
                                        )

            if (robot == 'pongbot_rbf' or 'pongbot'):                                        
                flip_sign_index = torch.tensor([1, 3])
    except:
        print("error get_num_legjoints function, please recheck the name of the robot")
    return num_legs, num_joints, motor_mapping, flip_sign_index

class RBFNet:
    def __init__(self, 
                 popsize, 
                 num_basis,
                 num_output,
                 robot,
                 motor_encode='semi-indirect'):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.architecture = [num_basis, num_output]
        self.popsize = popsize

        # Initialize CPG
        self.O = torch.Tensor([[0.0, 0.18]]).expand(popsize, 2).cuda()
        self.t, self.x, self.y, self.period = self.pre_compute_cpg()

        # Rbf network
        self.num_basis = num_basis
        self.num_output = num_output
        self.variance = 25.0
        self.phase = 0

        # Pre calculated rbf layers output 
        self.ci, self.cx, self.cy, self.rx, self.ry, self.KENNE = self.pre_rbf_centers(
            self.period, self.num_basis, self.x, self.y, self.pppppppppppppppppppppppppi)
        self.KENNE = self.KENNE.cuda()

        # Get number of legs, joints, and motor mapping from model --> sim robot
        self.num_legs, self.num_joints, motor_mapping, self.flip_sign_index = get_num_legjoints(robot)
        self.indices = motor_mapping.cuda()

        # initilize motor encoding type (weights, CPGs' phase)
        self.motor_encode = motor_encode # 'direct', 'indirect'
        if self.motor_encode == 'indirect':
            self.weights = torch.Tensor(popsize, num_basis, num_output//self.num_legs).uniform_(-0.1, 0.1).cuda()
            # Initilize phase of each CPG
            phase_2 = int(self.period//2)
            self.phase = torch.Tensor([0, phase_2])

        if self.motor_encode == 'semi-indirect':
            self.weights = torch.Tensor(popsize, num_basis, num_output//2).uniform_(-0.1, 0.1).cuda()
            # Initilize phase of each CPG
            phase_2 = int(self.period//2)
            self.phase = torch.Tensor([0, phase_2])

        step = self.num_joints
        self.indices_L = [(0, i * step) if i % 2 == 0 else (1, i * step) 
                          for i in range(self.num_legs//2)]
        self.indices_R = [(1, i * step) if i % 2 == 0 else (0, i * step) 
                          for i in range(self.num_legs//2)]


    def forward(self, pre):
        # print('pre: ', pre)
        
        with torch.no_grad():
            # Indirect encoding ##################################
            p1 = self.KENNE[int(self.phase[0])]
            p2 = self.KENNE[int(self.phase[1])]

            out_p1 = torch.tanh(torch.matmul(p1, self.weights))
            out_p2 = torch.tanh(torch.matmul(p2, self.weights))
            # Extend out_p1 from shape [3,3] to [3,6] by repeating along the last dimension
            # print('out_p1: ', out_p1.shape)
            out_p1 = out_p1.repeat(1, 2)
            out_p2 = out_p2.repeat(1, 2)
            outL, outR = self.concat_slices([out_p1, out_p2])
            # print('outR: ', outR)
            post = torch.concat([outL, outR], dim=1)
            # print('post: ', post.shape)
            post = torch.index_select(post, 1, self.indices)

            # Flip sign for specified indices
            if hasattr(self, 'flip_sign_index'):
                for idx in self.flip_sign_index:
                    post[:, idx] *= -1

            # print('post: ', post)
            # print('post index: ', post.shape)

            self.phase = self.phase + 1
            self.phase = torch.where(self.phase > self.period, 0, self.phase)
            ####################################################

        return post.float().detach()
    
    def concat_slices(self, tensors, dim=1):    
        outL = torch.cat([tensors[i][:, j:j+self.num_joints] for i, j in self.indices_L], dim=dim)
        outR = torch.cat([tensors[i][:, j:j+self.num_joints] for i, j in self.indices_R], dim=dim)
        return outL, outR    

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
            
    
    def pre_compute_cpg(self):
        # Run for one period
        phi   = 0.06*np.pi # SO(2) Frequency
        alpha = 1.01         # SO(2) Alpha term
        w11   = alpha*cos(phi)
        w12   = alpha*sin(phi)
        w21   =-w12
        w22   = w11
        x     = []
        y     = []
        t     = []
        t.append(0)
        x.append(-0.197)
        y.append(0.0)
        period = 0
        while y[period] >= y[0]:
            period = period+1
            t.append(period*0.0167)
            x.append(tanh(w11*x[period-1]+w12*y[period-1]))
            y.append(tanh(w22*y[period-1]+w21*x[period-1]))
            
        while y[period] <= y[0]:
            period = period+1
            t.append(period*0.0167)
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

        xy = x+y

        for k in range(len(ci)):
            cx[k] = x[ci[k]]
            cy[k] = y[ci[k]]

        for i in range(num_basis):
            rx   = [q - cx[i] for q in x]
            ry   = [q - cy[i] for q in y]
            KENNE[i] = np.exp(-(np.power((rx),2) + np.power((ry),2))*var)

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

####################################################