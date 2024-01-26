from agent import A2CAgent
from attention_model import AttentionModel
from env_tw import DataGenerator, Env
from baseline import RolloutBaseline as Baseline
import torch
args = {
    'n_epochs': 20,
    'n_batch': 3,
    'batch_size': 4,
    'n_nodes': 100,
    'initial_demand_size': 1,
    'max_load': 9,
    'speed': 1,
    'lambda': 1,
    'data_dir': 'datasets',
    'log_dir': 'logs',
    'save_path': 'saved_models_20',
    'decode_len': 20,   
    'actor_net_lr': 0.0001,
    'lr_decay': 1.0,
    'max_grad_norm': 1.0,
    'save_interval': 1,
    'bl_alpha': 0.05,
    'embedding_dim': 128,
    'random_seed': 1,
    'vehicle_num': 5,
    'early_coef': 0.1,
    'late_coef': 0.5
}
data_generator = DataGenerator(args)
#data = data_generator.get_test_all()
env = Env(args)
model = AttentionModel(args['embedding_dim'], args['embedding_dim'], args['n_nodes'])
agent = A2CAgent(model, args, env, data_generator)
baseline = Baseline(agent, agent.model, args, data_generator)
agent.train_epochs(baseline)


#print('loading...')
#d = torch.load('saved_models_20/best_model.pt')
#model.load_state_dict(d['model'])
#data = torch.load('C:/Users/User/Desktop/sdvrp_upd/VRP-size-512-len-20.txt')
#print(agent.rollout_test(data, model).mean())









