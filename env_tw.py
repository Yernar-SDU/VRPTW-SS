# from options import ParseParams
import numpy as np
import torch

class DataGenerator(object):
    def __init__(self, args):
        self.args = args
        self.rnd = np.random.RandomState(seed=args['random_seed'])
        # self.test_data = create_test_dataset(args)
      
    
    def get_train_next(self):
        args = self.args
        # coordinates = np.random.uniform(0, 100, size=(100, 100,2))
        # coordinates = torch.randint(low=0, high=100, size=(n_batches, batch_size, n_nodes, 2))
        # coordinates[0, 0]
        
        coordinates = torch.randint(low=0, high=100, 
                                    size=(self.args['n_batch'], self.args['batch_size'], self.args['n_nodes'] + 1, 2))
        coordinates[:,:,0,0] = 50
        coordinates[:,:,0,1] = 50
        demands = torch.normal(mean=15, std=10, 
                               size=(self.args['n_batch'], self.args['batch_size'],  self.args['n_nodes'] + 1, 1))
        demands = np.minimum(42, np.maximum(1, np.abs(torch.floor(demands))))
        demands[:, :, 0, :] = 0
        time_windows = self.generate_time_windows(coordinates)
        time_windows[:, :, 0, 0] = 0
        time_windows[:, :, 0, 1] = 1000
        
        torch.cat((demands, coordinates, time_windows), -1)
        return torch.cat((demands, coordinates, time_windows), -1)
        
        # Function to calculate L2 distance between depot and customer
    def calculate_distances_from_depot(self, coordinates):
        distances = torch.sqrt((torch.tensor(50) - coordinates[:,:,:,0])**2 + (torch.tensor(50) - coordinates[:,:,:,1])**2)
        return distances
     
    # Function to generate time windows for a customer
    def generate_time_windows(self, coordinates):
        b_0 = 1000
        
        d_0i = self.calculate_distances_from_depot(coordinates)
        h_hat_i = torch.ceil(d_0i) + 1
        a_sample = h_hat_i
        
        b_sample = b_0 - h_hat_i  # Assuming b_0 is defined
        
        # print(b_sample.shape)
        # print(a_sample.shape)
        a_i = torch.round((b_sample - a_sample)
                           * torch.rand((self.args['n_batch'], self.args['batch_size'], self.args['n_nodes'] + 1)) + a_sample)

        epsilon_hat = torch.randn(1)
        epsilon = max(abs(epsilon_hat), 1/100)
        epsilon
        
        b_i = np.minimum(torch.floor(a_i + 300 * epsilon), b_sample)
    
        a_i = a_i.unsqueeze(-1)  # Adds a new dimension at the end
        b_i = b_i.unsqueeze(-1)  # Adds a new dimension at the end
        result = torch.cat((a_i, b_i), dim=-1)
        return result
    
    
    
    
    
    
    
class Env(object):
    
    
    def __init__(self, args):
        self.args = args
        self.max_laod = args['max_load']
        self.n_nodes = args['n_nodes']
        self.batch_size = args['batch_size']
        self.vehicle_num = args['vehicle_num']
        self.speed = 1
        self.early_coef = args['early_coef']
        self.late_coef = args['late_coef']
        
        
    def reset(self, data):
        self.data = data
        
        
        self.coordinates = data[:, :, 1:3]
        self.demand = data[:,:, 0]
        self.time_windows = data[:, :, 3:]
        self.time_left = torch.zeros(self.batch_size, self.vehicle_num)
        
        
        
        self.cur_loads = torch.full((self.batch_size, self.vehicle_num, 1),
                                    self.max_laod, dtype=torch.int)
        
        
        
        self.dist_mat = torch.zeros(self.batch_size, self.n_nodes+1,
                                    self.n_nodes+1, dtype=torch.float)


        print('self.dist_mat.shape', self.dist_mat.shape)
        print('self.data.shape', self.data.shape)
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                self.dist_mat[:, i, j] = ((self.data[:, i, 0] - self.data[:, j, 0])**2 + (self.data[:, i, 1] - self.data[:, j, 1])**2)**0.5
                self.dist_mat[:, j, i] =  self.dist_mat[:, i, j]
        
        
        self.mask = torch.zeros(self.batch_size, self.vehicle_num, self.n_nodes + 1, dtype=torch.long)
        # self.mask_vehicle = torch.ones(self.batch_size, self.vehicle_num, dtype=torch.long)
        
        self.cur_time = torch.zeros(self.batch_size, self.vehicle_num, 1)
        self.cur_locs = torch.full((self.batch_size, self.vehicle_num, 1), 0)
        
        self.R = torch.zeros(self.batch_size, 1, dtype=torch.float)
        
        
        return data, self.mask, self.cur_locs, self.cur_loads, self.demand 
        # return self.state_v, self.state_d, self.mask_vehicle, self.mask_drone
    
    
    def estimate_nearest(self):
        self.cur_locs
    
    
    def step(self, idxs):
        new_locs = idxs
        # Calculate estimated time for all vehicles
        # Align new_locs to have the same structure as cur_locs
        new_locs = torch.stack(new_locs, dim=1).unsqueeze(-1)
        
        estimated_time = self.dist_mat[torch.arange(self.batch_size)[:, None, None], self.cur_locs, new_locs] / self.speed
        indices = torch.arange(new_locs.size(-1)).unsqueeze(0).unsqueeze(1)
        
        
        self.cur_time += estimated_time
        
        self.mask.scatter_(2, new_locs, 1)
        
        # for time in estimated_time:
          
        print(self.cur_time)
            
        # early_came_penalty = torch.max(0, self.time_windows[3, idxs] - self.cur_time, dim=1)  
        # late_came_penalty = torch.max(0, self.cur_time - self.time_windows[4], dim=1)
        # print(early_came_penalty)
        # print(late_came_penalty)
          
        self.R += (torch.sum(estimated_time, dim=1))
        print("R: ", self.R)
        finished =False
        return self.data, self.mask, self.cur_locs, self.cur_loads, self.demand, finished 
             
             
             
         
         
# args = ParseParams()   
# dataGen = DataGenerator(args)
# data = dataGen.get_train_next()
# env = Env(args)

# env.reset(data[0])


# idx = torch.tensor([1,2,3,4,5, 1, 2,3,4,5]).reshape(args['batch_size'], -1)
# env.step(idx)
# idx = torch.tensor([1,2,3,8,5, 1, 2,3,8,5]).reshape(args['batch_size'], -1)
# env.step(idx)
            