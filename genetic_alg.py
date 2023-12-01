

import EasyGA
import numpy as np
from math import pi

def generate_chromosome():
    l_bt = np.random.uniform(0,0.05)
    m_bt = np.random.uniform(0.03,0.07)

    l_th = np.random.uniform(0,100)
    m_th = np.random.uniform(60,140)
    h_th = np.random.uniform(100,200)

    theta_n = np.random.uniform(0,pi/3)

    ship_n = np.random.uniform(25,45)

    low_bullet_time = [0,0,l_bt]
    medium_bullet_time = [0,m_bt,0.1]
    # we set bullet_tim_high using smf
    low_thrust = [0,0,l_th]
    medium_thrust = [0,m_th,200]
    high_thrust = [h_th,200,200]


    NS_theta = [-pi/3,-theta_n,0]
    Z_theta = [-theta_n,0,theta_n]
    PS_theta = [0,theta_n,pi/3]

    turn_nl = [-180,-180,-ship_n]
    turn_ns = [-3*ship_n,-ship_n,0]
    turn_z = [-ship_n,0,ship_n]
    turn_ps = [0,ship_n,ship_n*3]
    turn_pl = [ship_n,180,180]
    chromosome = [
        low_bullet_time,
        medium_bullet_time,
        low_thrust,
        medium_thrust,
        high_thrust,
        NS_theta,
        Z_theta,
        PS_theta,
        turn_nl,
        turn_ns,
        turn_z,
        turn_ps,
        turn_pl
    ]
    return chromosome

def fitness():
    pass


def get_best_chromosome():
    pass
