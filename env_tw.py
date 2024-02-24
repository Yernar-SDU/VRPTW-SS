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
        self.cur_time = torch.zeros(self.batch_size, self.vehicle_num, 1)
        self.cur_locs = torch.full((self.batch_size, self.vehicle_num, 1), 0)
        
        self.R = torch.zeros(self.batch_size, 1, dtype=torch.float)
        
        
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
        
        estimated_time = self.dist_mat[torch.arange(self.batch_size)[:, None, None], self.cur_locs, new_locs] / self.speed
        # print('cur_locs', self.cur_locs)
        # print('new_locs', new_locs)
        # print('estimated_time', estimated_time)
        
        self.cur_time += estimated_time
        # print('new_locs', new_locs)
        #subtracting demand from served customers
        # Create a 1D index array
        idx_array = new_locs[:, :, 0]
        # Set demand to 0 for valid indices
        demand_values = self.demand[torch.arange(self.batch_size).view(-1, 1), idx_array]
        # print('demand', demand_values)
        # print('cur_laods', self.cur_loads)
        self.cur_loads -= demand_values.unsqueeze(-1)
        
        # print('cur_loads after.', self.cur_loads)
        # print('cur_loads', self.cur_loads)
        self.demand[torch.arange(self.batch_size).view(-1, 1), idx_array] = 0
        # print('demand after', self.demand)
        
        
        # print('self.cur_loads[:, vehicle_idx]', self.cur_loads[:, 0].shape)
        # print('self.demand', self.demand.shape)
        # print('self.demand!=0', (self.demand !=0).shape)
        # print('torch.where', torch.where(torch.logical_and(
        #     self.cur_loads[:, 1] >= self.demand, self.demand != 0), 0, 1))
        
        self.cur_locs = new_locs

        # Identify indices where self.cur_locs is 0 (depot)
        depot_indices = (self.cur_locs == 0)
        
        # Set the load back to 100 for vehicles at the depot
        self.cur_loads[depot_indices] = self.args['max_load']
        

        for vehicle_idx in range(self.vehicle_num):
            self.mask[:, vehicle_idx] = torch.where(torch.logical_and(
                self.cur_loads[:, vehicle_idx] >= self.demand, self.demand != 0), 0, 1)
        
        
        # print('mask2', self.mask)
            
        for batch in range(self.batch_size):
            for vehicle_idx in range(self.vehicle_num):
                if torch.all(self.mask[batch, vehicle_idx] ==1):
                    self.mask[batch, vehicle_idx, 0] = 0
        
        # Identify the positions where all values are 1
        # Check if all values are 1 except the first position
        
        # all_ones_except_first = torch.all(self.mask[:, :, 1:] == 1, dim=2)
        # print('koko', all_ones_except_first)
        # # Set the corresponding positions to 0 in the first position
        # self.mask[all_ones_except_first, 0] = 0   
            
        # print('mask2', self.mask)
        # print('demand2', self.demand)        
        # print('cur_locs', self.cur_locs)
        # print('cur_loads', self.cur_loads)
        # print('cur_time', self.cur_time)
        
        early_time_values = torch.stack([self.time_windows[i, idx, 0] for i, idx in enumerate(new_locs)], dim=0)
        late_time_values = torch.stack([self.time_windows[i, idx, 1] for i, idx in enumerate(new_locs)], dim=0)
        
        
        
        late_penalty = torch.maximum(self.cur_time - late_time_values,
                                    torch.zeros_like(self.cur_time - late_time_values)) 
        early_penalty = torch.maximum(early_time_values - self.cur_time,
                                    torch.zeros_like(early_time_values - self.cur_time))
        
        
        sum_early_penalty = torch.sum(early_penalty, dim=(1, 2), keepdim=True).squeeze().view(-1, 1)
        sum_late_penalty = torch.sum(late_penalty, dim=(1, 2), keepdim=True).squeeze().view(-1, 1)
        
        
        service_times = torch.where(torch.eq(new_locs, 0), torch.tensor(0), torch.tensor(10))
        # print('idxs', idxs)
        # print('cur_time1', self.cur_time)
        self.cur_time += service_times
        # print('cur_time2', self.cur_time)
        sum_service_times = torch.sum(service_times, dim=1)
        # print('service times', sum_service_times)
        self.R += (torch.sum(estimated_time, dim=1) + sum_early_penalty * self.early_coef + sum_late_penalty * self.late_coef)
        # print('R', self.R)
        self.R += sum_service_times
        # print('R', self.R)
        
        # print('sum_early_penalty', sum_early_penalty[0])
        # print('sum_late_penalty', sum_late_penalty[0])
        # print('estimated time ', (torch.sum(estimated_time[0], dim=1)))
        # print('R', self.R[0])
        
        if torch.all(self.demand == 0):
            finished = True                
        return self.data, self.mask, self.cur_locs, self.cur_loads, self.demand, finished 
             

# data_generator = DataGenerator(args)
# print('test', data_generator.get_test_next().shape)
# print('train', data_generator.get_train_next().shape)
#data = data_generator.get_test_next()

            