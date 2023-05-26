#encoding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import runpy
#import time as t
import os

import snp


## Fonctions ##########################################################################

def get_data(filename):
    time = []
    values = []
    with open(filename, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        header = True
        for row in spamreader:
            if header:
                header = False
            else:
                if str(row[0])[0:2] != ' [':
                    time.append(int(row[0]))
                    values.append(float(row[1]))
    return time, values

# Indicateur 1 : volume sous la courbe de masse de probabilité au dessus de la médiane

def indicator1(time, values, target, iters_max):
    surface = 0
    time_value_change = []

    last_change = values[0]
    time_value_change.append((time[0],last_change))

    for i in range(1,len(time)):
        if values[i] != last_change:
            last_change = values[i]
            time_value_change.append((time[i],last_change))

    j = 0
    while (j < len(time_value_change)-1):
        t1,v1 = time_value_change[j]
        t2,v2 = time_value_change[j+1]
        surface += (t2-t1) * abs(target-v1)
        j += 1

    if (t2 < iters_max-1):
        surface += (iters_max-1-t2) * abs(target-v2)

    return surface

# Indicateur 2 : list of 1 if value >= seuil and 0 otherwise

def indicator2(values, seuil):
    vector = np.array(values) >= seuil
    vector = 1*vector
    vector = vector.tolist()
    return vector

def indicator3(proba_values):
    aire = 0
    i = 0
    while i < len(proba_values)-1:
        aire += min(proba_values[i],proba_values[i+1]) # aire du rectangle
        if proba_values[i] != proba_values[i+1]:
            aire += abs(proba_values[i+1] - proba_values[i])/2# aire du triangle
        i += 1
    return aire

# Sample from algorithm and compute indicators

def compute_indictors(param_pb,nb_sampling_algo,seuil):
    volume = 0
    matrix = []
    target = param_pb['target']
    algo = param_pb['algo']
    print('Algo :', algo)
    iters_max = param_pb['iters_max']

    for i in range(nb_sampling_algo):
        # launch optimisation algorithm
        snp.run(param_pb)

        # get data
        time, values = get_data(algo+".csv")
        
        if (algo == 'num_evolution_strategies' or algo == 'bit_evolution_strategies'):
            time_modif = []
            values_modif = []
            i = 0
            t = 1
            val = 0
            while t <= min(time[-1],iters_max):
                if t == time[i]:
                    val = values[i]
                    i += 1
                values_modif.append(val)
                time_modif.append(t)
                t += 1
            time = time_modif
            values = values_modif
                    
        # indicator 1
        #volume += indicator1(time, values, target, iters_max)
        # indicator 2
        matrix.append(indicator2(values, seuil))

    # indicator 1
    #volume /= target ## peut-etre à diviser par 30*30=surface max
    #print("\n Indicateur 1 : " + str(volume))

    # indicator 2

    for vector in matrix: # prolongement des probas obtenus jusqu'à nb iters max
        while len(vector) < iters_max:
            vector.append(vector[-1])

    proba_values = np.zeros(iters_max)
    for vector in matrix:
        proba_values = np.add(proba_values,vector)
    proba_values /= nb_sampling_algo
    
    print(" Indicateur 2 : " + str(proba_values[-1]))
    
    # indicator 3
    aire = indicator3(proba_values)
    print(" Indicateur 3 : " + str(aire))

    return (aire, proba_values)


## Comparison of 2 instances of algorithm ########################################

compute_comparison_2_algo = False

if compute_comparison_2_algo:

    # Paramètres
    
    param = {}
    param1 = {}
    param2 = {}
    
    # Definition du problème
    param['nb_sensors'] = 3
    param['sensor_range'] = 0.3
    param['domain_width'] = 30
    param['target'] = param['domain_width']*param['domain_width']
    
    # Paramètres des algorithmes
    param['temperature_init'] = 1000 # 1
    param['beta'] = 3 # 1/0.995
    param['nb_pop'] = 6
    
    # Critères d'arrêt
    param['iters_max'] = 100 # budget
    param['steady_delta'] = 50
    param['steady_epsilon'] = 0
    param['variation_scale'] = 0.3
    
    # Paramètres des indicateurs
    nb_sampling_algo = 3
    param['seed'] = None
    seuil = 650 # 668
    
    # Choix des algorithmes
    param1 = param.copy()
    param2 = param.copy()
    param1['algo'] = "num_recuit_simule"
    param2['algo'] = "num_evolution_strategies"
    
    
    # Sampling of 2 algorithms and compute indicators
    
    (aire_1, proba_values_1) = compute_indictors(param1,nb_sampling_algo,seuil)
    (aire_2, proba_values_2) = compute_indictors(param2,nb_sampling_algo,seuil)
    
    plt.figure()
    plt.plot(proba_values_1, color='red', label=param1['algo'])
    plt.plot(proba_values_2, color='green', label=param2['algo'])
    plt.legend()
    plt.title("Indicateur 2 : coupe dans l'EAF selon le seuil=" + str(seuil))
    plt.ylabel('probabilités')
    plt.xlabel("nombre d'appels à la fonction objectif")
    plt.show()


## Design of experiment #######################################################

compute_design_of_experiment = True
compute_ERS_recuit_simule = False
compute_ERS_evolution_strategies = False

if compute_design_of_experiment:

    # Paramètres fixes

    param = {}
    param['domain_width'] = 30
    param['target'] = param['domain_width']*param['domain_width']
    param['iters_max'] = 100 # budget
    param['steady_delta'] = 50
    param['steady_epsilon'] = 0
    param['variation_scale'] = 0.3
    nb_sampling_algo = 5 
    param['seed'] = None
    param['temperature_init'] = None
    param['beta'] = None
    param['nb_pop'] = None
    seuil = 660
    seuil_prop = 0.8
    
    # Paramètres à faire varier
    
    nb_sensors_D = [7, 10, 13, 15]
    sensor_range_D = [0.1, 0.15, 0.2]
    #domain_width_D = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    temperature_init_D = [50, 100]
    beta_D = [2, 3, 4, 5]
    nb_pop_D = [5, 6, 7, 8]

# Algorithme recuit simule

if compute_ERS_recuit_simule == True:

    param1 = param.copy()
    param1['algo'] = "num_recuit_simule"
    
    best_temperature_init = None
    best_beta = None
    best_aire_1 = 0
    
    plt.figure()
    
    # Définition de l'instance d'algoritme
    for temperature_init in temperature_init_D:
        param1['temperature_init'] = temperature_init
        for beta in beta_D:
            param1['beta'] = beta
            mean_proba_values_1 = np.zeros(param['iters_max'])
            compt_pb = 0
            
            # Définition du problème
            for nb_sensors in nb_sensors_D:
                param1['nb_sensors'] = nb_sensors
                for sensor_range in sensor_range_D:
                    param1['sensor_range'] = sensor_range
                    
                    target_coverage = min(param1['domain_width']**2, np.pi*(sensor_range*param1['domain_width'])**2*nb_sensors)
                    seuil = seuil_prop * target_coverage
                    
                    # Calcul des indicateurs
                    (aire_1, proba_values_1) = compute_indictors(param1,nb_sampling_algo,seuil)
                    mean_proba_values_1 += proba_values_1
                    compt_pb += 1
            
            mean_proba_values_1 /= compt_pb
            
            # Affichage de l'ERS
            plt.plot(mean_proba_values_1, label='T0='+str(temperature_init)+' et beta='+str(beta))
                    
                    
    plt.legend()
    plt.title("Moyenne des ERS selon des seuils à " + str(seuil_prop*100) + "% de la couverture target")
    plt.ylabel('probabilités')
    plt.xlabel("nombre d'appels à la fonction objectif")
    plt.show()
    

# Algorithme génétique

if compute_ERS_evolution_strategies:

    param2 = param.copy()
    param2['algo'] = "num_evolution_strategies"
                
    best_nb_pop = None
    best_moy_aire_2 = 0
    
    plt.figure()
    
    # Définition de l'instance d'algoritme
    for nb_pop in nb_pop_D:
        param2['nb_pop'] = nb_pop
        mean_proba_values_2 = np.zeros(param['iters_max'])
        compt_pb = 0
        
        # Définition du problème
        for nb_sensors in nb_sensors_D:
            param2['nb_sensors'] = nb_sensors
            for sensor_range in sensor_range_D:
                param2['sensor_range'] = sensor_range
                
                target_coverage = min(param2['domain_width']**2, np.pi*(sensor_range*param2['domain_width'])**2*nb_sensors)
                seuil = seuil_prop * target_coverage
                
                # Calcul des indicateurs
                (aire_2, proba_values_2) = compute_indictors(param2,nb_sampling_algo,seuil)
                mean_proba_values_2 += proba_values_2
                compt_pb += 1
        
        mean_proba_values_2 /= compt_pb
        
        # Affichage de l'ERS
        plt.plot(mean_proba_values_2, label='nb_pop='+str(nb_pop))
                
                    
    plt.legend()
    plt.title("Moyenne des ERS selon des seuils à " + str(seuil_prop*100) + "% de la couverture target")
    plt.ylabel('probabilités')
    plt.xlabel("nombre d'appels à la fonction objectif")
    plt.show()
    