import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import ERL_model
from mlp import MLPIndiv
from neuroevolution import SSNE
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env

import warnings
warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("\nPyTorch device:", device.type, "(" + str(torch.cuda.device_count()) + ")\n")
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        print("\nPyTorch device:", device.type, "(" + str(os.cpu_count()) + ")\n")

    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)
    

    print("-",str(num_jobs)+str(num_mas).zfill(2),"-\n")

    env_valid = get_validate_env(env_valid_paras)  # Create an environment for validation
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')


    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time[:-2])
    os.makedirs(save_path)


    pop_size = model_paras["pop_size"]
    n_hidden_actor = model_paras["n_hidden_actor"]
    actor_dim = model_paras["actor_in_dim"]
    n_latent_actor = model_paras["n_latent_actor"]
    action_dim = model_paras["action_dim"]


    # Popopulation-based
    evolver = SSNE(pop_size)
    gen_no = model_paras["gen_no"]  # Generation number
    elite_fraction = model_paras["elite_frac"] # Fraction of elite individuals
    crossover_prob = model_paras["cx_rate"] # Crossover rate
    mutation_prob = model_paras["mt_rate"] # Mutation rate


    # Start training iteration
    start_time = time.time()
    env = None

    pop = []
    for __ in range(pop_size):
        pop.append(MLPIndiv(n_hidden_actor, actor_dim, n_latent_actor, action_dim).to(device))
    
    result_makespans = []

    for i in range(1, gen_no+1):

        # Replace training instances every x iteration (x = 20 in paper)
        if (i - 1) % train_paras["parallel_iter"] == 0:
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            print('\nnum_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        print("\nGEN_" + str(i))
        all_fitness = []
        all_makespans = []
        test_makespans = []

        last_time = time.time()
        
        for id in range(1, pop_size+1):
            actor = pop[id-1]
            memories = ERL_model.Memory()
            model = ERL_model.ERL(model_paras, train_paras, actor, num_envs=env_paras["batch_size"])

            # Get state and completion signal
            state = env.state
            done = False
            dones = env.done_batch

            # Schedule in parallel
            fitness = 0
            with torch.no_grad():
                while ~done:
                    actions = model.policy_old.act(state, memories, dones)
                    state, rewards, dones = env.step(actions)
                    fitness += rewards
                    
                    done = dones.all()
                    memories.rewards.append(rewards)
                    memories.is_terminals.append(dones)
                    # gpu_tracker.track()  # Used to monitor memory (of gpu)

            # Scores
            fitness = fitness.mean()
            all_fitness.append(fitness.mean().item())
            
            makespan = env.makespan_batch.mean()
            all_makespans.append(makespan.item())

            # print("I_" + str(id) + " (fitness:", '%.3f' % fitness + "; makespan:", '%.3f' % makespan + ")")   

            # Asign fitness
            actor.fitness = fitness
            memories.clear_memory()
            env.reset()
            
            # Record the average results and the results on each instance
            test_makespan = validate(env_valid_paras, env_valid, model.policy_old)
            test_makespans.append(test_makespan.item())

            # Save the best model
            if test_makespan < makespan_best:
                makespan_best = test_makespan
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_gen({3})_id({4}).pt'.format(save_path, num_jobs, num_mas, i, id)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)
        
        train_time = time.time()-last_time
        print("train_time: {:.3f}s".format(train_time))

        print("")
        print("Fitness    (Avg:", '%.3f' % np.mean(all_fitness), "| Max:", '%.3f' % np.max(all_fitness), "| Min:", '%.3f' % np.min(all_fitness) + ")")
        print("Makespan   (Avg:", '%.3f' % np.mean(all_makespans), "| Max:", '%.3f' % np.max(all_makespans), "| Min:", '%.3f' % np.min(all_makespans) + ")")
        print("Validation (Avg:", '%.3f' % np.mean(test_makespans), "| Max:", '%.3f' % np.max(test_makespans), "| Min:", '%.3f' % np.min(test_makespans) + ")")

        # Save the best result
        result_makespans.append(np.min(test_makespans))

        # Evolution
        evolver.epoch(i, pop, all_fitness)


    # Save the data of training curve to files
    ss = str(num_jobs)+"_"+str(num_mas)

    data_avg = pd.DataFrame([np.arange(1, gen_no+1, 1), result_makespans]).T
    data_avg.to_csv('{0}/training_avg_{1}.csv'.format(save_path, ss),
                    header=['Iter', 'Min Makespan'], index=False)

    print("\ntotal_time: {:.3f}s".format(time.time()-start_time))

if __name__ == '__main__':
    main()



