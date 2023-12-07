import os
import time

from EasyGA import EasyGA
from kesslergame import KesslerController, Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt
import time

from skfuzzy.control import ControlSystem

from test_controller import TestController


class GeneticController(KesslerController):
        
    def __init__(self,chromosome=None):
        super().__init__()  # Call the parent class constructor if necessary
        self.eval_frames = 0
        if chromosome is None:
            self.chromosome = self.optimize().gene_value_list[0]
        else:
            if type(chromosome) == EasyGA.Chromosome:
                self.chromosome = chromosome.gene_value_list[0]
            else:
                self.chromosome = chromosome
        self.action_control = self.setup_action_control()



    def setup_action_control(self):
        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare

        print(self.chromosome)

        bullet_s = self.chromosome[0]
        bullet_m = self.chromosome[1]
        thrust_l = self.chromosome[2]
        thrust_m = self.chromosome[3]
        thrust_h = self.chromosome[4]
        theta_ns = self.chromosome[5]
        theta_z = self.chromosome[6]
        theta_ps = self.chromosome[7]
        turn_nl = self.chromosome[8]
        turn_ns = self.chromosome[9]
        turn_z = self.chromosome[10]
        turn_ps = self.chromosome[11]
        turn_pl = self.chromosome[12]

        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        ship_thrust = ctrl.Consequent(np.arange(0, 200, 5), 'ship_thrust')

        chromosome = self.chromosome

        # TODO: use chromosome here to adjust the membership functions for all of these.
        # chromosome generation based on membership functions seen here
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,bullet_s)
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, bullet_m)
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

        ship_thrust['Low'] = fuzz.trimf(ship_thrust.universe, thrust_l)
        ship_thrust['Medium'] = fuzz.trimf(ship_thrust.universe, thrust_m)
        ship_thrust['High'] = fuzz.trimf(ship_thrust.universe, thrust_h)

        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3,-1*math.pi/6)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, theta_ns)
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, theta_z)
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, theta_ps)
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,math.pi/6,math.pi/3)

        #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, turn_nl)
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, turn_ns)
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, turn_z)
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, turn_ps)
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, turn_pl)
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rules = [
            ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'], ship_thrust['High'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NL'], ship_fire['N'], ship_thrust['High'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'], ship_thrust['Medium'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PL'], ship_fire['N'], ship_thrust['High'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'], ship_thrust['High'])),  
            ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'], ship_thrust['High'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['Medium'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'], ship_thrust['Medium'])),    
            ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['Medium'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'], ship_thrust['High'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'], ship_thrust['Medium'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['Medium'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'], ship_thrust['Low'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['Medium'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'], ship_thrust['Medium'])),
        ]
        
        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
             
        action_control = ctrl.ControlSystem(rules)

        return action_control


    def optimize(self):

        database_path = '/content/database.db'
        journal_path = '/content/database.db-journal'
        if os.path.exists(database_path):
            # If it exists, delete the file
            os.remove(database_path)
        if os.path.exists(journal_path):
            # If it exists, delete the file
            os.remove(journal_path)
        ga = EasyGA.GA()
        ga.gene_impl = lambda: generate_chromosome()
        ga.chromosome_length = 1
        ga.population_size = 1
        ga.target_fitness_type = 'max'
        ga.generation_goal = 1
        ga.fitness_function_impl = fitness
        ga.evolve()
        ga.print_best_chromosome()
        y = ga.population[0]
        
        return y

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        vel = ship_state['speed']
        heading = ship_state['heading']

        print("Ship Velocity %f" %vel)
        print("Ship is heading %f" %heading)

        # if vel == 0:
        #     vel = 0+0.000000000001

        # if heading == 90:
        #     heading == 90-0.00000000001


        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - (bullet_speed+vel)**2) * (closest_asteroid["dist"])**2)
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -(bullet_speed+vel)**2))
        intrcpt2 = ((-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-(bullet_speed+vel)**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * bullet_t
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * bullet_t
        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        # Pass the inputs to the rulebase and fire it
        print(isinstance(self.action_control, ControlSystem))
        print(self.action_control)
        shooting = ctrl.ControlSystemSimulation(self.action_control,flush_after_run=1)
        
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        
        shooting.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
               
        # And return your three outputs to the game simulation. Controller algorithm complete.
        dist_from_ctrl1 = math.sqrt((ship_pos_x - 400)**2 + (ship_pos_y - 400)**2)
        if dist_from_ctrl1 < 50:
            thrust = -3*(shooting.output['ship_thrust'])
        else:
            if bullet_t < 0.1:
                thrust = -3*(shooting.output['ship_thrust'])
            else:
                thrust = shooting.output['ship_thrust']
            
            self.eval_frames +=1
        
        #DEBUG
        print(thrust, bullet_t, shooting_theta, turn_rate, fire)

        return thrust, turn_rate, fire

    @property
    def name(self) -> str:
        return "Genetic Controller"


def fitness(chromosome):
    my_test_scenario = Scenario(name='Test Scenario', num_asteroids=5,
                                ship_states=[{'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1},
                                             {'position': (600, 400), 'angle': 90, 'lives': 3, 'team': 2},
                                             ],
                                map_size=(1000, 800),
                                time_limit=60,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)
    game_settings = {'perf_tracker': True,
                     'graphics_type': GraphicsType.Tkinter,
                     'realtime_multiplier': 1,
                     'graphics_obj': None}
    game = TrainerEnvironment(settings=game_settings)
    controller = GeneticController(chromosome)
    score, perf_data = game.run(scenario=my_test_scenario, controllers=[TestController(), controller])

    return score.teams[1].accuracy


def generate_chromosome():
    # bullet time ranges
    l_bt = np.random.uniform(0, 0.05)
    m_bt = np.random.uniform(0.03, 0.07)

    # thrust ranges
    l_th = np.random.uniform(0, 100)
    m_th = np.random.uniform(60, 140)
    h_th = np.random.uniform(100, 200)

    # theta number
    theta_n = np.random.uniform(0, math.pi / 3)

    # ship
    ship_n = np.random.uniform(25, 45)

    low_bullet_time = [0, 0, l_bt]
    medium_bullet_time = [0, m_bt, 0.1]
    # we set bullet_tim_high using smf

    # thrust mem functions
    low_thrust = [0, 0, l_th]
    medium_thrust = [0, m_th, 200]
    high_thrust = [m_th, 200, 200]


    # theta mem functions
    NS_theta = [-math.pi / 3, -theta_n, 0]
    Z_theta = [-theta_n, 0, theta_n]
    PS_theta = [0, theta_n, math.pi / 3]

    # ship turn mem functions
    turn_nl = [-180, -180, -ship_n]
    turn_ns = [-3 * ship_n, -ship_n, 0]
    turn_z = [-ship_n, 0, ship_n]
    turn_ps = [0, ship_n, ship_n * 3]
    turn_pl = [ship_n, 180, 180]

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
