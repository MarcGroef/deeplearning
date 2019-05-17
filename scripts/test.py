#!/usr/bin/python
import numpy as np
from trainer import Trainer

def runDropout():
    nExps = 10
    results = np.zeros((mExps))

    for expIdx in range(nExps):
        print "Experiment index = ", expIdx
        exp = {'exp_type' : 'dropout', 'fc_dropout_rate' : 0.4, 'nExperiments' : nExps, 'experiment_index': expIdx}
        trainer = Trainer('dropout', **exp)
        trainer.train()
        results[expIdx] = float(trainer.test())
    
    print "Result Dropout: ", np.mean(results), "+-", np.std(results)
    
    ##Results: acc = 
    return results
    
def runBatchnorm():
    nExps = 10
    results = np.zeros((mExps))

    for expIdx in range(nExps):
        print "Experiment index = ", expIdx
        exp = {'exp_type' : 'batchnorm', 'nExperiments' : nExps, 'experiment_index': expIdx}
        trainer = Trainer('batchnorm', **exp)
        trainer.train()
        results[expIdx] = float(trainer.test())
    
    print "Result Batchnorm: ", np.mean(results), "+-", np.std(results)
    ##Result: acc = 
    return results
    
def runControl():
    nExps = 10
    results = np.zeros((mExps))
    
    for expIdx in range(nExps):
        print "Experiment index = ", expIdx
        exp = {'exp_type' : 'control', 'nExperiments' : nExps, 'experiment_index': expIdx}
        trainer = Trainer('control', **exp)
        trainer.train()
        results[expIdx] = float(trainer.test())
    
    print "Result control: ", np.mean(results), "+-", np.std(results)
    ##Result: acc =
    return results
    
if __name__ == "__main__":
    dropout = runDropout()
    batchnorm = runBatchnorm()
    control = runControl()
    print "Result Dropout: ", np.mean(dropout), "+-", np.std(dropout)
    print "Result Batchnorm: ", np.mean(batchnorm), "+-", np.std(batchnorm)
    print "Result control: ", np.mean(control), "+-", np.std(control)

    