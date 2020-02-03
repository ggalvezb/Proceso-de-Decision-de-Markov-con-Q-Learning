# C:\Users\ggalv\Google Drive\Respaldo\Magister\Modelos estocasticos\Trabajo 1				python clasificador.py

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import copy

#Lectura de datos----------------------------------------------------------------------------
def lectura_datos():
	direccion_ola = pd.read_excel('C:\\Users\\ggalv\\Google Drive\\Respaldo\\Magister\\Modelos estocasticos\\Trabajo 1\\6_direccion_ola.xlsx', sheet_name='Hoja1')
	direccion_ola = direccion_ola.as_matrix()


	metros_ola = pd.read_excel('C:\\Users\\ggalv\\Google Drive\\Respaldo\\Magister\\Modelos estocasticos\\Trabajo 1\\6_metros_ola.xlsx', sheet_name='Hoja1')
	metros_ola = metros_ola.as_matrix()


	periodo_ola = pd.read_excel('C:\\Users\\ggalv\\Google Drive\\Respaldo\\Magister\\Modelos estocasticos\\Trabajo 1\\6_periodo_ola.xlsx', sheet_name='Hoja1')
	periodo_ola = periodo_ola.as_matrix()

	return(direccion_ola,metros_ola,periodo_ola)

#Clasificacion de estados--------------------------------------------------------------------
def clasificacion_estados(periodo_ola,metros_ola,direccion_ola):
	estados=[]
	for i in range(364):
		for j in range(6):
			if direccion_ola[i][j]==0:		#Si la direccion de la ola no es favorable se asigna una estrella a ese caso
				estados.append(1)
			else:							#Si la direccion de la ola es favorable, se evaluan los otros parametros y se asignan las estrellas correspondientes
				if 	metros_ola[i][j]>=2.5 and periodo_ola[i][j]>=13:
					estados.append(5)
				elif metros_ola[i][j]>=2.5 and periodo_ola[i][j]<13 and periodo_ola[i][j]>=10:
					estados.append(4)
				elif metros_ola[i][j]>=2.5 and periodo_ola[i][j]<10:	
					estados.append(2)
				elif metros_ola[i][j]<2.5 and metros_ola[i][j]>=1.5 and periodo_ola[i][j]>=13:
					estados.append(4)		
				elif metros_ola[i][j]<2.5 and metros_ola[i][j]>=1.5 and periodo_ola[i][j]>=10 and periodo_ola[i][j]<13:
					estados.append(4)
				elif metros_ola[i][j]<2.5 and metros_ola[i][j]>=1.5 and periodo_ola[i][j]<10:
					estados.append(2)
				elif metros_ola[i][j]<1.5 and periodo_ola[i][j]>=13:
					estados.append(3)
				elif metros_ola[i][j]<1.5 and periodo_ola[i][j]<13 and periodo_ola[i][j]>=10:
					estados.append(2)
				elif metros_ola[i][j]<1.5 and periodo_ola[i][j]<10:
					estados.append(1)					
	return(estados)

#Matriz de frecuencia------------------------------------------------------------------------
def matriz_frecuencia(estados):
	matriz_frecuencia=np.zeros((n_states,n_states))		#Creo una matriz vacia de ceros y dimension 5 por 5
	for i in range(len(estados) - 1):		#incremento el valor de acuerdo a las transiciones de los estados
		matriz_frecuencia[estados[i] -1 ][estados[i+1] - 1]=matriz_frecuencia[estados[i] -1 ][estados[i+1] - 1]+1
	# print("matriz_frecuencia =",matriz_frecuencia)
	return(matriz_frecuencia)

#Matriz de probabilidades--------------------------------------------------------------------
def matriz_probabilidades(matriz_frecuencia,n_states):
	matriz_probabilidades=np.zeros((n_states,n_states))
	for i in range(n_states):
		for j in range(n_states):
			matriz_probabilidades[i][j]=(int(matriz_frecuencia[i][j]) / int(sum(matriz_frecuencia[i])))
	# print(matriz_probabilidades)
	return(matriz_probabilidades)	

n_states=5
direccion_ola,metros_ola,periodo_ola=lectura_datos()
estados=clasificacion_estados(periodo_ola,metros_ola,direccion_ola)
matriz_frecuencia=matriz_frecuencia(estados)	
matriz_probabilidades=matriz_probabilidades(matriz_frecuencia,n_states)
print("matriz_probabilidades=",matriz_probabilidades)


######################################################
# ## Markov Decision Processes------------------------
######################################################

#Matriz de probabilidades para proceso de decision-------------------------------------------
def transition_probabilities(matriz_probabilidades,n_states):
	matriz_probabilidades=matriz_probabilidades.tolist() 
	transition_probabilities=[]
	for i in range(n_states):
		transition_probabilities.append(list([list(matriz_probabilidades[i]),list(matriz_probabilidades[i])]))
	#print(transition_probabilities)
	print("transition_probabilities=",transition_probabilities)		
	return(transition_probabilities)


#Matriz de rewards --------------------------------------------------------------------------
def rewards(n_states):
	restar=5
	quedarse=[]
	for j in range(n_states):
		temp=[]
		for i in range(n_states):
			temp.append(n_states-restar)
			restar-=1
		quedarse.append(temp)
		restar=5
		restar+=j + 1
	irse=copy.deepcopy(quedarse)
	for k in range(len(irse)):
		for l in range(len(irse)):
			irse[k][l]=irse[k][l]*-1
	rewards=[]
	
	for m in range(len(quedarse)):
		temp2=[]
		temp2.append(quedarse[m])
		temp2.append(irse[m])
		rewards.append(temp2)			
	return(rewards)		



#Main----------------------------------------------------------------------------------------	

possible_actions=[[1,0], [0, 1], [0, 1], [0, 1], [0, 1]]
transition_probabilities=transition_probabilities(matriz_probabilidades,n_states)
rewards=rewards(n_states)
print(rewards)
def policy_random(state):
    return np.random.choice(possible_actions[state]) # random actions

class MDPEnvironment(object):
    def __init__(self, num_states=5, start_state=2):
        self.start_state=start_state
        self.num_states=num_states
        self.reset()
    def reset(self):
        self.total_rewards = 0
        self.state = self.start_state
    def step(self, action):
        next_state = np.random.choice(range(self.num_states), p=transition_probabilities[self.state][action]) # Select next state according to action and transition probability
        reward = rewards[self.state][action][next_state] #R(s,a,s')
        self.state = next_state # set state as next_state
        self.total_rewards += reward # increment reward
        return self.state, reward


def run_episode(policy, n_steps, num_states=5, start_state=2, display=True):
    env = MDPEnvironment(num_states)
    if display:
        print("States (+rewards):", end=" ")
    for step in range(n_steps):
        if display:
            if step == 10:
                print("...", end=" ")
            elif step < 10:
                print(env.state, end=" ")
        action = policy(env.state)
        state, reward = env.step(action)
        if display and step < 10:
            if reward:
                print("({})".format(reward), end=" ")
    if display:
        print("Total rewards =", env.total_rewards)
    return env.total_rewards            

#Q-Learning ---------------------------------------------------------------------------------
n_actions = 2
n_steps = 100
alpha = 0.01
gamma = 0.99
exploration_policy = policy_random
q_values = np.full((n_states, n_actions), -np.inf) # Initialize Q_k(s',a') as -inf
for state, actions in enumerate(possible_actions):
    q_values[state][actions]=0 # Initialize only possible Q_k(s',a') as 0

env = MDPEnvironment() # Create a MDP environment
for step in range(n_steps):
    action = exploration_policy(env.state) #np.random.choice(possible_actions[state])
    state = env.state 
    next_state, reward = env.step(action) # set next state and reward for the action taken
    next_value = np.max(q_values[next_state]) # greedy policy
    q_values[state, action] = (1-alpha)*q_values[state, action] + alpha*(reward + gamma * next_value) # update q-values according to the Q-Learning algorithm
    
def optimal_policy(state):
    return np.argmax(q_values[state])

all_totals = []
for episode in range(7000):
    all_totals.append(run_episode(optimal_policy, n_steps=100, display=(episode<10)))
print("Summary: mean={:.1f}, std={:1f}, min={}, max={}".format(np.mean(all_totals), np.std(all_totals), np.min(all_totals), np.max(all_totals)))
print()


print("Que hacer en el estado 1=",optimal_policy(0))
print("Que hacer en el estado 2=",optimal_policy(1))
print("Que hacer en el estado 3=",optimal_policy(2))
print("Que hacer en el estado 4=",optimal_policy(3))
print("Que hacer en el estado 5=",optimal_policy(4))



#[[[1, 0, 0, 0, 0, 0]],[[1, 0, 0, 0, 0, 0], [0, 0.8333333333333334, 0.007575757575757576, 0.007575757575757576, 0.12121212121212122, 0.030303030303030304]], [[1, 0, 0, 0, 0, 0], [0, 0.015873015873015872, 0.6825396825396826, 0.031746031746031744, 0.25396825396825395, 0.015873015873015872]], [[1, 0, 0, 0, 0, 0], [0, 0.0, 0.08333333333333333, 0.75, 0.16666666666666666, 0.0]], [[1, 0, 0, 0, 0, 0], [0, 0.01549053356282272, 0.012908777969018933, 0.0051635111876075735, 0.8950086058519794, 0.07142857142857142]], [[1, 0, 0, 0, 0, 0], [0, 0.0037974683544303796, 0.0012658227848101266, 0.0, 0.10506329113924051, 0.889873417721519]]]


