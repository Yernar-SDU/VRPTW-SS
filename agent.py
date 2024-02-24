import torch
import torch.optim as optim
import os
import time
import math
import matplotlib.pyplot as plt
from torch.nn import DataParallel
import numpy as np
import sys
import copy

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class State(object):

    def __init__(self, mask, cur_loc, cur_load, demand):
        self.demand = demand
        self.mask = mask
        self.cur_load = cur_load.to(device)
        self.cur_loc = cur_loc.to(device)

    def __getitem__(self, item):
        return {
            'mask': self.mask[item],
            'cur_loc': self.cur_loc[item],
            'cur_load': self.cur_load[item],
            'demand': self.demand[item]
        }

    def update(self, mask, cur_loc, cur_load, demand):
        self.demand = demand
        self.mask = mask
        self.cur_load = cur_load.to(device)
        self.cur_loc = cur_loc.to(device)
        

    def update_mask(self ,idx):
        
        for batch in range(self.mask.shape[0]):
            if idx[batch] != 0:
                self.mask[batch, idx] = 1
                
            if torch.all(self.mask[batch]==1):
                self.mask[batch, 0] = 0

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class A2CAgent(object):

    def __init__(self, model, args, env, dataGen, test_data):
        self.model = model
        self.args = args
        self.env = env
        self.dataGen = dataGen
        self.test_data = test_data
        # Initialize optimizer
        self.optimizer = optim.Adam([{'params': model.parameters(), 'lr': args['actor_net_lr']}])
        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: args['lr_decay'] ** epoch)
        out_file = open(os.path.join(args['log_dir'], 'results.txt'), 'w+')
        print("agent is initialized")

    def train_epochs(self, baseline):
        args = self.args
        model = self.model
        test_rewards = []
        best_model = 100000
        losses = []

        start_time = time.time()
        reward_epoch = torch.zeros(args['n_epochs'])
        train_rewards = []
        for epoch in range(args['n_epochs']):
            # [batch_size, n_nodes, 3]: entire epoch train data
            train_data = self.dataGen.get_train_next()
            
            # compute baseline value for the entire epoch
            baseline_data = baseline.wrap_dataset(train_data)
            # compute for each batch the rollout
            # train each batch
            # train_rewards = []
            for batch in range(args['n_batch']):
                print("batch: ", batch)
                print("epoch: ", epoch)
                # evaluate b_l with  new train data and old model
                data, bl_val = baseline.unwrap_batch(baseline_data[batch])
                bl_val = move_to(bl_val, device) if bl_val is not None else None
                R, logs, actions = self.rollout_train(data)
                # train_rewards.append(R.mean())
                # Calculate loss
                adv = (R - bl_val).to(device)
                # print('R', R)
                # print('bl_val', bl_val)
                loss = (adv * logs).mean()
                # print('loss', loss)
                losses.append(loss)
                # Perform backward pass and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # print('asdf', self.model.state_dict())
                # Clip gradient norms and get (clipped) gradient norms for logging
                grad_norms = clip_grad_norms(self.optimizer.param_groups, args['max_grad_norm'])
                self.optimizer.step()
            epoch_duration = time.time() - start_time
            print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
            
            # avg_reward = torch.mean(self.rollout_test(self.test_data[0], self.model).to(torch.float32))
            # print("average test reward: ", avg_reward)
            # reward_epoch[epoch] = avg_reward
            
            
            if (epoch % args['save_interval'] == 0) or epoch == args['n_epochs'] - 1:
                print('Saving model and state...')
                torch.save(
                    {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': baseline.state_dict()
                    },
                    os.path.join(args['save_path'], 'epoch-{}.pt'.format(epoch))
                )
            # if avg_reward < best_model:
            #     best_model = avg_reward
            #     torch.save(
            #         {
            #             'model': self.model.state_dict(),
            #             'optimizer': self.optimizer.state_dict(),
            #             'rng_state': torch.get_rng_state(),
            #             'cuda_rng_state': torch.cuda.get_rng_state_all(),
            #             'baseline': baseline.state_dict()
            #         },
            #         os.path.join(args['save_path'], 'best_model.pt')
                # )
                
            test_success, train_reward = baseline.epoch_callback(self.model, epoch, self.test_data)
            train_rewards.append(train_reward)
            if test_success:
                # self.test_data = self.dataGen.get_test_next()
                baseline._update_model(model, epoch, self.test_data)
                torch.save(
                    {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': baseline.state_dict()
                    },
                    os.path.join(args['save_path'], 'best_model.pt')
                    )
            # test_rewards.append(avg_reward.cpu().numpy())
            np.savetxt(args['save_path']+"/test_rewards.txt", train_rewards)
            # np.savetxt("trained_models/losses.txt", losses)
            # lr_scheduler should be called at end of epoch
            # self.lr_scheduler.step()
            # ratio = reward_epoch/total_demand_epoch
            
            plt.plot(torch.arange(len(train_rewards)), train_rewards)
            plt.savefig(os.path.join('plots', f'train_rewards-{self.env.n_nodes}.png'))
            
            
            plt.plot(torch.arange(len(test_rewards)), test_rewards)
            plt.savefig(os.path.join('plots', f'test_rewards-{self.env.n_nodes}.png'))
            
            
            
        plt.plot(torch.arange(args['n_epochs']).numpy(), reward_epoch.cpu().numpy())
        plt.savefig(os.path.join('plots', f'reward_ratio-{self.env.n_nodes}.png'))

    def rollout_train(self, data):
        env = self.env
        model = self.model
        model.train()
        set_decode_type(self.model, "sampling")
        data = copy.deepcopy(data)

        print('train data', data.shape)
        data, masks, cur_locs, cur_loads, demand = env.reset(data)
        
        data = move_to(data, device)
        
        vehicle_states = []
        
        for vehicle in range(env.vehicle_num):
            state = State(masks[:, vehicle], cur_locs[:, vehicle], 
                          cur_loads[:, vehicle], demand)
            vehicle_states.append(state) 
        
        embeddings, fixed = model.embed(data)
        
        
        # print("{}: {}".format("initial state", state[0]))
        # print("{}: {}".format("mask", mask[0]))
        logs = []
        actions = []
        time_step = 0

        while time_step < self.args['decode_len']:
            indexes = []
            # indexes = move_to(indexes, device)
            # print('time_step', time_step)
            for vehicle in range(env.vehicle_num):
                # print('mask4', vehicle_states[vehicle].mask)
                log_p, idx = model(embeddings, fixed, vehicle_states[vehicle])
                logs.append(log_p[:, 0, :])
                # print('idx', idx)
                indexes.append(idx.cpu())
                # print('idxese', indexes)
                actions.append(idx)
                
                for vehicle in range(env.vehicle_num):
                    vehicle_states[vehicle].update_mask(idx)
                
                
                
                
            time_step += 1
                
            data, mask, cur_locs, cur_loads, demand, finished = env.step(indexes)
            if finished:
                break
            
            data = move_to(data, device)
            for vehicle in range(env.vehicle_num):    
                vehicle_states[vehicle].update(mask[:, vehicle], cur_locs[:, vehicle], cur_loads[:, vehicle], demand)
            # embeddings, fixed = model.embed(data)
             
        # print("{}: {}".format("state update", state[0]))
        # print("{}: {}".format("mask", mask[0]))
        # print("{}: {}".format("state", env.state[0]))

        R = (env.R).to(device)
        print("R: ", torch.mean(R.to(torch.float32)))
        logs = torch.stack(logs, 1)
        actions = torch.stack(actions, 1)

        logs = model._calc_log_likelihood(logs, actions)

        return R, logs, actions

    def rollout_test(self, data_o, model):
        env = self.env
        model.eval()
        set_decode_type(self.model, "greedy")
        
        print('test data', data_o.shape)
        data, masks, cur_locs, cur_loads, demand = env.reset(data_o)
        data = move_to(data, device)
        vehicle_states = []
        

        for vehicle in range(env.vehicle_num):
            state = State(masks[:, vehicle], cur_locs[:, vehicle], 
                          cur_loads[:, vehicle], demand)
            vehicle_states.append(state) 
        
        embeddings, fixed = model.embed(data)
        # print('kkkk', data[:, :, :3].shape)
        # print('koko', mask.shape)
        
        # print("{}: {}".format("initial state", state[0]))
        # print("time_demand: ", env.time_demand[0])
        time_step = 0

        while time_step < self.args['decode_len']:
            
            indexes = []
            # indexes = move_to(indexes, device)
            # print('time_step', time_step)
            for vehicle in range(env.vehicle_num):
                # print('vehicle', vehicle)
                # print('vehicle_state', vehicle_states[vehicle])
                # print('vehicle load', vehicle_states[vehicle].cur_load.shape)
                
                
                log_p, idx = model(embeddings, fixed, vehicle_states[vehicle])
                # print('idx', idx)
                indexes.append(idx.cpu())
                # print('idxese', indexes)
                for vehicle in range(env.vehicle_num):
                    vehicle_states[vehicle].update_mask(idx)
                
                # torch.cat((indexes, idx.unsqueeze(0)), dim=0)
                
                # print(time_step, " time step")
                
            # print('indexes', indexes)
            # print("{}: {}".format("state update", state[0]))
            # print("cur_time", env.cur_time[0])
            # print("time_demand: ", env.time_demand[0])
            # print("R: ", env.reward[0])
            time_step += 1
            # indexes = torch.tensor(indexes)
            data, mask, cur_locs, cur_loads, demand, finished = env.step(indexes)
            if finished:
                break
            data = move_to(data, device)
            for vehicle in range(env.vehicle_num):    
                vehicle_states[vehicle].update(mask[:, vehicle], cur_locs[:, vehicle], cur_loads[:, vehicle], demand)
            # embeddings, fixed = model.embed(data)
            
            
            
        R = torch.mean(env.R).to(device)
            
        return R
    
        
