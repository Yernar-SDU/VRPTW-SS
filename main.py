from agent import A2CAgent
from attention_model import AttentionModel
from env_tw_2 import DataGenerator, Env
from baseline import RolloutBaseline as Baseline
import pandas as pd
import torch
args = {
    'n_epochs': 100,
    'n_batch': 20,
    'batch_size': 4,
    'n_nodes': 20,
    'initial_demand_size': 1,
    'max_load': 50,
    'speed': 1,
    'lambda': 1,
    'data_dir': 'datasets',
    'log_dir': 'logs',
    'save_path': 'saved_models_20',
    'decode_len': 200,   
    'actor_net_lr': 0.1,
    'lr_decay': 0.0001,
    'max_grad_norm': 2.0,
    'save_interval': 1,
    'bl_alpha': 0.05,
    'embedding_dim': 128,
    'random_seed': 1234,
    'vehicle_num': 2,
    'early_coef': 0.1,
    'late_coef': 0.5,
    'val_size': 4,
    'test_b_size': 4
}


data_generator = DataGenerator(args)
env = Env(args)
model = AttentionModel(args['embedding_dim'], args['embedding_dim'], args['n_nodes'])
# d = torch.load('saved_models_20/best_model.pt')
# model.load_state_dict(d['model'])
test_data = data_generator.get_test_next()
agent = A2CAgent(model, args, env, data_generator, test_data)
baseline = Baseline(agent, agent.model, args, data_generator, test_data)

agent.train_epochs(baseline)

# data = data_generator.get_test_next()
# print(data[0][0])

# data_test = torch.load('testing_data.pth')
# print(agent.rollout_test(data_test, model))







