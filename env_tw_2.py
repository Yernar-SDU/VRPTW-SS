import pandas as pd
import numpy as np
import torch
import sys
class DataGenerator(object):
    def __init__(self, args):
        self.args = args
        # self.rnd = np.random.RandomState(seed=args['random_seed'])
        # self.test_data = create_test_dataset(args)
        # torch.manual_seed(args['random_seed'])
        # np.random.seed(args['random_seed'])
      
    
    def get_train_next(self):
        args = self.args
        # coordinates = np.random.uniform(0, 100, size=(100, 100,2))
        # coordinates = torch.randint(low=0, high=100, size=(n_batches, batch_size, n_nodes, 2))
        # coordinates[0, 0]
        
        coordinates = torch.randint(low=0, high=100, 
                                    size=(self.args['n_batch'], self.args['batch_size'], self.args['n_nodes'] + 1, 2))
        coordinates[:,:,0,0] = 40
        coordinates[:,:,0,1] = 50
        demands = torch.normal(mean=15, std=10, 
                               size=(self.args['n_batch'], self.args['batch_size'],  self.args['n_nodes'] + 1, 1))
        demands = np.minimum(42, np.maximum(1, np.abs(torch.floor(demands))))
        demands[:, :, 0, :] = 0
        time_windows = self.generate_time_windows(coordinates)
        time_windows[:, :, 0, 0] = 0
        time_windows[:, :, 0, 1] = 1000
        
        # print('coordinates', coordinates)
        # print('demands', demands.shape)
        # print('time_windows', time_windows.shape)
        
        # torch.cat((demands, coordinates, time_windows), -1)

        return torch.cat((demands, coordinates, time_windows), -1)
    
    def get_test_next(self):
        args = self.args
        # coordinates = np.random.uniform(0, 100, size=(100, 100,2))
        # coordinates = torch.randint(low=0, high=100, size=(n_batches, batch_size, n_nodes, 2))
        # coordinates[0, 0]
        
        coordinates = torch.randint(low=0, high=100, 
                                    size=(self.args['test_b_size'], self.args['batch_size'], self.args['n_nodes'] + 1, 2))
        coordinates[:,:,0,0] = 40
        coordinates[:,:,0,1] = 50
        demands = torch.normal(mean=15, std=10, 
                                size=(self.args['test_b_size'], self.args['batch_size'],  self.args['n_nodes'] + 1, 1))
        demands = np.minimum(42, np.maximum(1, np.abs(torch.floor(demands))))
        demands[:, :, 0, :] = 0
        time_windows = self.generate_time_windows_test(coordinates)
        time_windows[:, :, 0, 0] = 0
        time_windows[:, :, 0, 1] = 1000
        
        # print('coordinates', coordinates.shape)
        # print('demands', demands.shape)
        # print('time_windows', time_windows.shape)
        
        # torch.cat((demands, coordinates, time_windows), -1)
        return torch.cat((demands, coordinates, time_windows), -1)
    
    # def get_test_new(self):
    #     args = self.args
    #     # coordinates = np.random.uniform(0, 100, size=(100, 100,2))
    #     # coordinates = torch.randint(low=0, high=100, size=(n_batches, batch_size, n_nodes, 2))
    #     # coordinates[0, 0]
        
    #     coordinates = torch.randint(low=0, high=100, 
    #                                 size=(self.args['n_batch'], self.args['batch_size'], self.args['n_nodes'] + 1, 2))
    #     coordinates[:,:,0,0] = 50
    #     coordinates[:,:,0,1] = 50
    #     demands = torch.normal(mean=15, std=10, 
    #                            size=(self.args['n_batch'], self.args['batch_size'],  self.args['n_nodes'] + 1, 1))
    #     demands = np.minimum(42, np.maximum(1, np.abs(torch.floor(demands))))
    #     demands[:, :, 0, :] = 0
    #     time_windows = self.generate_time_windows(coordinates)
    #     time_windows[:, :, 0, 0] = 0
    #     time_windows[:, :, 0, 1] = 1000
        
    #     return torch.cat((demands, coordinates, time_windows), -1)
            
    
    
    
    
    # def get_test_next(self):
    #     path = r'C:\Users\user3\Desktop\AkhmetbekYernar\research\solomon_dataset\RC2'
    #     data = pd.read_csv(path + '\RC201.csv')
    #     # print(data)    
    #     coordinates =torch.unsqueeze(torch.tensor(data[['XCOORD.', 'YCOORD.']].values, dtype=torch.float32), dim=0)
    #     demands = torch.unsqueeze(torch.tensor(data['DEMAND']), dim=0)
    #     demands = demands.unsqueeze(2)
    #     time_windows = torch.unsqueeze(torch.tensor(data[['READY TIME', 'DUE DATE']].values), dim=0)
    #     # print('coordinates', coordinates.shape)
    #     # print('demands', demands.shape)
    #     # print('time_windows', time_windows.shape)
    #     # print('hello', torch.cat((demands, coordinates, time_windows), -1))
    #     return torch.cat((demands, coordinates, time_windows), -1).repeat(self.args['batch_size'],1,1).unsqueeze(0).repeat(25, 1, 1, 1)
    
    
        
        
        
        # Function to calculate L2 distance between depot and customer
    def calculate_distances_from_depot_test(self, coordinates):
        distances = torch.sqrt((torch.tensor(40) - coordinates[:,:,:,0])**2 + (torch.tensor(50) - coordinates[:, :,:,1])**2)
        return distances
    
    def generate_time_windows_test(self, coordinates):
        b_0 = 1000
        
        d_0i = self.calculate_distances_from_depot_test(coordinates)
        h_hat_i = torch.ceil(d_0i) + 1
        a_sample = h_hat_i
        
        b_sample = b_0 - h_hat_i  # Assuming b_0 is defined
        
        # print(b_sample.shape)
        # print(a_sample.shape)
        a_i = torch.round((b_sample - a_sample)
                           * torch.rand((self.args['test_b_size'], self.args['batch_size'], self.args['n_nodes'] + 1)) + a_sample)

        epsilon_hat = torch.randn(1)
        epsilon = max(abs(epsilon_hat), 1/100)
        epsilon
        
        b_i = np.minimum(torch.floor(a_i + 300 * epsilon), b_sample)
    
        a_i = a_i.unsqueeze(-1)  # Adds a new dimension at the end
        b_i = b_i.unsqueeze(-1)  # Adds a new dimension at the end
        result = torch.cat((a_i, b_i), dim=-1)
        return result
    
    
    
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
        # print('asl', data.shape)
        
        self.coordinates = data[:, :, 1:3]
        self.demand = data[:,:, 0]
        self.time_windows = data[:, :, 3:]
        self.time_left = torch.zeros(self.batch_size, self.vehicle_num)
        # print('coordinates', self.coordinates.shape)
        
        self.cur_loads = torch.full((self.batch_size, self.vehicle_num, 1),
                                    self.max_laod, dtype=torch.float)
        
        self.left_time = torch.zeros(self.batch_size, self.vehicle_num, 1)
        
        
        self.dist_mat = torch.zeros(self.batch_size, self.n_nodes+1,
                                    self.n_nodes+1, dtype=torch.float)


        # print('self.dist_mat.shape', self.dist_mat.shape)
        # print('self.data.shape', self.data.shape)
        for i in range(self.n_nodes + 1):
            for j in range(i+1, self.n_nodes + 1):
                self.dist_mat[:, i, j] = ((self.data[:, i, 0] - self.data[:, j, 0])**2 + (self.data[:, i, 1] - self.data[:, j, 1])**2)**0.5
                self.dist_mat[:, j, i] =  self.dist_mat[:, i, j]
        
        # print('dist_mat', self.dist_mat)
        
        self.mask = torch.zeros(self.batch_size, self.vehicle_num, self.n_nodes + 1, dtype=torch.long)
        # self.mask_vehicle = torch.ones(self.batch_size, self.vehicle_num, dtype=torch.long)
        self.mask[:, :, 0] = 1
        self.cur_time = torch.zeros(self.batch_size)
        self.cur_locs = torch.full((self.batch_size, self.vehicle_num, 1), 0)
        
        self.R = torch.zeros(self.batch_size, dtype=torch.float)
        
        
        return data, self.mask, self.cur_locs, self.cur_loads, self.demand 
        # return self.state_v, self.state_d, self.mask_vehicle, self.mask_drone
    
    
    def estimate_nearest(self):
        self.cur_locs
    
        
    
    def step(self, idxs):
        
        # print('idxs', idxs[0][0], idxs[1][0])
        # print('cur_loads', self.cur_loads[0])
        # print('mask1', self.mask[0])
        # print('demand', self.demand[0])
        # print('cur_time', self.cur_time[0])
        
        finished = False
        
        new_locs = torch.stack(idxs, dim=1).unsqueeze(-1)
        print('------------------STARTING------------')
        print('cur_locs', self.cur_locs)
        print('new_locs', new_locs)
        print('self.cur_loads', self.cur_loads)
        print('demand', self.demand)
        # Create a mask for changed locations
        changed_mask = self.cur_locs != new_locs
        
        # Convert the mask to integers (1 for True, 0 for False)
        changed_indices = changed_mask.int()
        
        estimated_time = self.dist_mat[torch.arange(self.batch_size)[:, None, None], self.cur_locs, new_locs] / self.speed
        
        self.left_time[changed_indices.bool()] = (estimated_time[changed_indices.bool()] + 10)
        
        print('left_time', self.left_time)  
        min_arrival_time, _ = torch.min(self.left_time, dim=1, keepdim=True)
        print('min_arrival_time', min_arrival_time)
        self.cur_time += min_arrival_time.squeeze(2).squeeze(1)
        
        self.left_time -= min_arrival_time
        print('self.left_time', self.left_time)
        
        served_vehicles = self.left_time == 0
        served_indexes= new_locs[served_vehicles]
        served_demands = self.demand[torch.arange(self.demand.size(0)), served_indexes]
        self.cur_loads[served_vehicles] = self.cur_loads[served_vehicles] - served_demands
        self.demand[torch.arange(self.demand.size(0)), served_indexes] = 0
           
        # Vectorized version
        for batch in range(self.batch_size):
            for vehicle in range(self.vehicle_num):
                if not served_vehicles[batch, vehicle]:
                    self.mask[batch, vehicle, :] = 1
                    self.mask[batch, vehicle, new_locs[batch, vehicle]] = 0
                    # print(self.mask[batch, vehicle])
                else:
                    self.mask[batch, vehicle] = torch.where(
                        torch.logical_and(self.cur_loads[batch, vehicle]
                                          > self.demand[batch], self.demand[batch] != 0), 0, 1)
                    for locs in new_locs:
                        self.mask[batch, vehicle, locs] = 1
                    
                    
                    if torch.all(self.mask[batch,vehicle] == 1):
                        self.mask[batch,vehicle, 0] = 0
                print('mask', self.mask[batch, vehicle])

        self.cur_locs = new_locs
        # Identify indices where self.cur_locs is 0 (depot)
        depot_indices = (self.cur_locs == 0)
        
        # Set the load back to max load for vehicles at the depot
        self.cur_loads[depot_indices] = self.args['max_load']
                    
        early_time_values = torch.stack([self.time_windows[i, idx, 0] for i, idx in enumerate(new_locs)], dim=0)
        late_time_values = torch.stack([self.time_windows[i, idx, 1] for i, idx in enumerate(new_locs)], dim=0)
        early_time_values = early_time_values[served_vehicles]
        late_time_values = late_time_values[served_vehicles]
                
        # Calculate penalties
        late_penalty = torch.clamp(self.cur_time - late_time_values, min=0)
        early_penalty = torch.clamp(early_time_values - self.cur_time, min=0)

        print('late_penalty', late_penalty)
        print('early_penalty', early_penalty)
        
        print('cur_time', self.cur_time)
        self.R += (early_penalty * self.early_coef) + (late_penalty * self.late_coef)
        self.R += min_arrival_time.squeeze(2).squeeze(1)
        print('self.R', self.R)
        print('------------------ENDINGG------------')
        
        if torch.all(self.demand == 0):
            finished = True                
        return self.data, self.mask, self.cur_locs, self.cur_loads, self.demand, finished 
      