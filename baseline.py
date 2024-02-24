import copy
import torch
from scipy.stats import ttest_rel


def rollout(agent, model, dataset, args, test=False):
    def eval_model_bat(bat):
        # evaluate the model
        cost = agent.rollout_test(bat, model)
        return cost

    bl_val = torch.zeros([args['n_batch'], args['batch_size']])

        
    dataset = copy.deepcopy(dataset)
    print("n_batch: ", args['n_batch'])
    # print("dataset: ", dataset.shape)
      
    
    if test:
        bl_val = torch.zeros([args['test_b_size'], args['batch_size']])
        for batch in range(args['test_b_size']):
            cost = eval_model_bat(dataset[batch])
            bl_val[batch, :] = cost
            print("batch b_l: ", batch, 'cost', cost)
                
    else:
        for batch in range(args['n_batch']):
            cost = eval_model_bat(dataset[batch])
            bl_val[batch, :] = cost
            print("batch b_l: ", batch, 'cost', cost)
              
    return bl_val


class BaselineDataset(object):

    def __init__(self, dataset=None, baseline=None):
        self.dataset = dataset
        self.baseline = baseline
        assert (self.dataset.shape[0] == self.baseline.shape[0])

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }


class RolloutBaseline(object):

    def __init__(self, agent, model, args, dataGen, test_data, epoch=0):
        self.model = model
        self.agent = agent
        self.dataGen = dataGen
        self.args = args
        self.test_data = test_data
        self._update_model(model, epoch, test_data)

    def _update_model(self, model, epoch, dataset=None, pretending_bl_vals=None):
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        print('datasetlialia', )
        if dataset is not None:
            print('dataset shakulia', dataset.shape)
            if dataset.shape[1] != self.args['val_size']:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
        #   elif (dataset[0] if self.problem.NAME == 'tsp' else dataset[0]['loc']).size(0) != self.opts.graph_size:
        #      print("Warning: not using saved baseline dataset since graph_size does not match")
        #      dataset = None
        # self.bl_vals = rollout(self.agent, self.model, dataset, self.args)

        if dataset is None:
            self.dataset = self.dataGen.get_train_next()
        else:
            self.dataset = dataset
            
        if pretending_bl_vals is not None:
            self.bl_vals = pretending_bl_vals
        else:
            self.bl_vals = rollout(self.agent, self.model, dataset, self.args, True)
            
        print("Evaluating baseline model on evaluation dataset")
        print("eval data: ", self.dataset.shape)
        self.mean = self.bl_vals.mean()
        print('epoch', epoch, 'mean R updated model:', self.mean)
       
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print("Evaluating baseline on training dataset...")

        return BaselineDataset(dataset, rollout(self.agent, self.model, dataset, self.args))

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline']  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.agent.rollout(x)

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch, dataset):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        
       
        test_success = False
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(self.agent, model, dataset, self.args, True)
        pretending_bl_vals = candidate_vals
        candidate_mean = candidate_vals.mean()
        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(    
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))

        if epoch %3 != 0:
            return False, candidate_mean

        if candidate_mean - self.mean < 0:
            # Calc p value
            n_batch, batch_size = candidate_vals.shape
            candidate_vals = candidate_vals.reshape(n_batch * batch_size)
            bl_vals = self.bl_vals.reshape(n_batch * batch_size)
            t, p = ttest_rel(candidate_vals, bl_vals)
            p_val = p / 2  # one-sided
            # print("Candidate values:", candidate_vals)
            # print("Baseline values:", bl_vals)
            # print("Candidate values are constant:", candidate_vals.std() == 0)
            # print("Baseline values are constant:", bl_vals.std() == 0)
            candidate_vals = candidate_vals[~torch.isnan(candidate_vals)]
            bl_vals = bl_vals[~torch.isnan(bl_vals)]
            assert len(candidate_vals) >= 3 and len(bl_vals) >= 3, "Insufficient data for the t-test"

            print("T-statistic: {}, P-value: {}".format(t, p))
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.args['bl_alpha']:
                
                if epoch % 3 == 0 or epoch > 2:
                    print('Update baseline')
                    test_success = True
        return test_success, candidate_mean
    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        load_model.load_state_dict(state_dict['model'].state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])
