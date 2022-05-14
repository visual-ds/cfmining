import pandas as pd
import yaml

def save_result(values, dataset, experiment, moo, show=True):
    assert dataset in  ['german', 'giveme', 'taiwan', 'student', 'pima']
    assert experiment in ['logistic', 'tree', 'monotone-tree']
    assert moo in ['MPC cost', 'MPC vs. #changes', 'feature']
    
    if show:
        print(yaml.dump(values, default_flow_style=False))
    
    with open('results/cache/'+'_'.join([dataset, experiment, moo])+'.yaml', 'w') as outfile:
        yaml.dump(values, outfile, default_flow_style=False)
        
        
def open_result(dataset, experiment, moo):
    with open('results/cache/+''_'.join([dataset, experiment, moo])+'.yaml', 'r') as f:
        data = yaml.load_all(f, Loader=yaml.SafeLoader)
    return data
    
    
    
    