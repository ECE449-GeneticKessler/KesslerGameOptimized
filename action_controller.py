import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

def setup_action_controller(chromosome):

    # determines the threat bearing etc where we shoot
    threat = ctrl.Antecedent(np.linspace(0, 10, 11), 'threat')

    # determines where we shoot or not
    target = ctrl.Antecedent(np.linspace(0, 10, 11), 'target')

    # determines how quickly we move to shoot
    turnrate = ctrl.Consequent(np.arange(-180,180,1), 'turn') # Degrees due to Kessler

    # determines whether we shoot or not (essentially equivalent of target)
    fire = ctrl.Consequent(np.arange(-1,1,0.1), 'fire')

    # determines if we move or not
    thrust = ctrl.Consequent(np.linspace(0, 10, 11), 'thrust')



    # insert rule set here after we determine the chromosome


    rules = []

    action_ctrl = ctrl.ControlSystem(rules)

    action_sim = ctrl.ControlSystemSimulation(action_ctrl)

    return action_sim


