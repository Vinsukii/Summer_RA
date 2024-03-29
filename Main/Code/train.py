import sys
import copy
import json
import os
import random
import time
import threading
from collections import deque
import utils.erl_utils as utils

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import ERL_model, PPO_model
from mlp import MLPIndiv
from neuroevolution import SSNE
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env

import warnings
warnings.filterwarnings('ignore')

# Multi-threading class for ERL2.1
class Thread(threading.Thread):
    def __init__(self, ID, id, actor, data):
        threading.Thread.__init__(self)
        self.ID = ID
        self.actor_id = id
        self.actor = actor
        self.data = data

        self.mk_best = None
        self.drl_actor = None
        self.drl_critic = None

    def run(self):
        print('\nStarting...({0})'.format(self.ID))
        mk_best, actor, critic = drl_training(self.ID, self.actor_id, self.actor, self.data)

        self.mk_best = mk_best
        self.drl_actor = actor
        self.drl_critic = critic
        print('Exiting...({0})\n'.format(self.ID))

# Function for performing DRL training step
def drl_training(T_id, a_id, actor, data):
        T_id = T_id              ; drl_best_models = data[6] 
        indv_id = a_id           ; drl_best_models_2 = data[7]
        env_paras = data[0]      ; drl_makespan_best = data[8][T_id-1]
        model_paras = data[1]    ; save_path = data[9]  
        train_paras = data[2]    ; save_path_nn = data[10]                         
        env_valid_paras = data[3]; model_path_nn = data[11] 
        env_valid = data[4]      ; last_time = data[12] 
        best_critic = data[5]    ; i = data[13]
                   
                 
        drl_losses = []
        drl_makespans = []
        drl_test_makespans = []

        env_valid = get_validate_env(env_valid_paras)

        num_jobs = env_paras["num_jobs"]
        num_mas = env_paras["num_mas"]
        opes_per_job_min = int(num_mas * 0.8)
        opes_per_job_max = int(num_mas * 1.2)

        best_actor = actor
        best_critic = best_critic

        if T_id == 0: best_critic = best_critic 
        else: best_critic = copy.deepcopy(best_critic)

        memories_ = PPO_model.Memory()
        model_ = PPO_model.PPO(model_paras, train_paras, best_actor, best_critic, num_envs=env_paras["batch_size"])

        print("\nA2C Training ({0})...".format(indv_id+1))
        
        if train_paras["sync_iter"] == model_paras["gen_no"]: max_iter = train_paras["max_iters"]
        else:
            if i == train_paras["sync_iter"]: max_iter = int(train_paras["max_iters"]*0.2)
            if i == train_paras["sync_iter"]*2: max_iter = int(train_paras["max_iters"]*0.8)
        
        for i_ in range(0, max_iter):
            # Replace training instances every x iteration (x = 20 in paper)
            if i_ % train_paras["parallel_iter"] == 0:
                nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
                case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
                env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
                print('\nnum_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

            drl_start_time = time.time()

            state = env.state
            done = False
            dones = env.done_batch

            while ~done:
                with torch.no_grad():
                    actions_ = model_.policy_old.act(state, memories_, dones)
                state, rewards, dones = env.step(actions_)
                done = dones.all()
                memories_.rewards.append(rewards)
                memories_.is_terminals.append(dones)
            
            train_time = time.time()-drl_start_time

            drl_mk = env.makespan_batch.mean()
            loss, reward = model_.update(memories_, env_paras, train_paras)

            drl_losses.append(loss)
            drl_makespans.append(drl_mk)
            
            env.reset()
            memories_.clear_memory()

            print(str(i_) + " (loss: ", '%.3f' % loss + "; reward: ", '%.3f' % reward + "; makespan: ", '%.3f' % drl_mk + ")")


            if i_ % train_paras["save_timestep"] == 0:
                drl_test_mk = validate(env_valid_paras, env_valid, model_.policy_old)
                drl_test_makespans.append(drl_test_mk.item())

                # Save the best model
                if drl_test_mk < drl_makespan_best:
                    drl_makespan_best = drl_test_mk
                    if len(drl_best_models) == 3:
                        delete_file = drl_best_models.popleft()
                        os.remove(delete_file)       

                    save_file = '{0}/save_best_{1}_{2}_gen({3})_id({4})_nn({5})_r({6})_t({7}).pt'.format(save_path_nn, num_jobs, num_mas, i, indv_id+1, i_, round(drl_test_mk.item(),2), T_id)
                    save_file_2 = '{0}/save_best_{1}_{2}_{3}.pt'.format(model_path_nn, num_jobs, num_mas, T_id)
                    
                    drl_best_models.append(save_file)

                    save_model = model_.policy.state_dict()
                    torch.save(save_model, save_file_2)
                    torch.save(save_model, save_file)
        

        print("("+str(T_id)+") ")
        print("Loss       (Avg:", '%.3f' % np.mean(drl_losses), "| Max:", '%.3f' % np.max(drl_losses), "| Min:", '%.3f' % np.min(drl_losses) + ")")
        print("Makespan   (Avg:", '%.3f' % np.mean(drl_makespans), "| Max:", '%.3f' % np.max(drl_makespans), "| Min:", '%.3f' % np.min(drl_makespans) + ")")
        print("Validation (Avg:", '%.3f' % np.mean(drl_test_makespans), "| Max:", '%.3f' % np.max(drl_test_makespans), "| Min:", '%.3f' % np.min(drl_test_makespans) + ")")

        drl_time = time.time()-last_time
        print("train_time: {:.3f}s".format(drl_time))


        # Save the data of training curve to files
        ss = str(num_jobs)+"_"+str(num_mas)

        data_avg = pd.DataFrame([np.arange(10, max_iter+10, 10), drl_test_makespans]).T
        data_avg.to_csv('{0}/training_avg_{1}_i({2}).csv'.format(save_path, ss, int(i/train_paras["sync_iter"])),
                        header=['Iter', 'Avg Makespan'], index=False)

        return drl_makespan_best, model_.policy.actor, model_.policy.critic


def find_ids(pop):
    dist_sum = [0,0,0,0,0,0,0,0,0,0]
    for indv in pop: dist_sum += indv.sum_params()[0]
    
    dist_avg = np.array(dist_sum/len(pop))

    # print(dist_avg)

    dist_ls = []
    for indv in pop: dist_ls.append(np.mean((dist_avg - indv.sum_params()[0])**2))

    return np.argsort(dist_ls)[:3]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# MAIN
def main(seed=0, method="", size="", algo=""):
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)

    #SEED
    if seed > 0: setup_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("\nPyTorch device:", device.type, "(" + str(torch.cuda.device_count()) + ")")
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        print("\nPyTorch device:", device.type, "(" + str(os.cpu_count()) + ")")
    print("Train seed:", seed, "\n")

    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    config_str = './configs/{0}/config_{1}.json'.format(size, algo)
    
    with open(config_str, 'r') as load_f:
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
    envS = str(num_jobs)+str(num_mas).zfill(2)

    print("-",envS,"-")

    env_valid = get_validate_env(env_valid_paras)  # Create an environment for validation
    maxlen = 1  # Save the best model
    best_models = deque()
    best_models_2 = deque()
    drl_best_models = deque()
    drl_best_models_2 = deque()
    makespan_best = float('inf')
    drl_makespan_best = [float('inf'),float('inf'),float('inf')]


    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    if train_paras["grid_runs"]:
        save_path = '../Results/{0}/{1}/grid/train_{2}/run_{3}'.format(method, envS, algo, seed)
    else:
        save_path = '../Results/{0}/{1}/save/train_{2}_{3}'.format(method, envS, algo, str_time[:-2])

    save_path_ne = save_path+'/ne'
    save_path_nn = save_path+'/nn'

    model_path = '../Results/{0}/{1}/model/{2}/'.format(method, envS, algo)
    model_path_ne = model_path+'/ne'
    model_path_nn = model_path+'/nn'

    os.makedirs(save_path)
    os.makedirs(save_path_ne)
    os.makedirs(save_path_nn)
    
    if seed == 1:
        os.makedirs(model_path)
        os.makedirs(model_path_ne)
        os.makedirs(model_path_nn)


    pop_size = model_paras["pop_size"]
    n_hidden_actor = model_paras["n_hidden_actor"]
    actor_dim = model_paras["actor_in_dim"]
    n_latent_actor = model_paras["n_latent_actor"]
    action_dim = model_paras["action_dim"]


    # Popopulation-based
    gen_no = model_paras["gen_no"]  # Generation number
    elite_fraction = model_paras["elite_frac"] # Fraction of elite individuals
    crossover_prob = model_paras["cx_rate"] # Crossover rate
    mutation_prob = model_paras["mt_rate"] # Mutation rate

    evolver = SSNE(pop_size, elite_fraction, crossover_prob, mutation_prob)

    best_critic = None


    # Start training iteration
    start_time = time.time()
    env = None

    pop = []
    for __ in range(pop_size):
        pop.append(MLPIndiv(n_hidden_actor, actor_dim, n_latent_actor, action_dim).to(device))

    best_id = 0
    best_actor = None
    result_makespans = []

    nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
    case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
    env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
    print('\nnum_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

    for i in range(1, gen_no+1):

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
                    actions = model.policy_old.act(state, memories, dones, False)
                    state, rewards, dones = env.step(actions)
                    fitness += rewards
                    
                    done = dones.all()
                    memories.rewards.append(rewards)
                    memories.is_terminals.append(dones)
                    # gpu_tracker.track()  # Used to monitor memory (of gpu)

            # Scores
            fitness = fitness.mean()
            all_fitness.append(fitness.item())
            
            makespan = env.makespan_batch.mean()
            all_makespans.append(makespan.item())  

            # Assign fitness
            actor.fitness = fitness
            memories.clear_memory()
            env.reset()
            
            # Record the average results and the results on each instance
            test_makespan = validate(env_valid_paras, env_valid, model.policy_old)
            test_makespans.append(test_makespan.item())

            # print("I_" + str(id) + " (fitness:", '%.3f' % fitness + "; makespan:", '%.3f' % makespan + "; vali:", '%.3f' % test_makespan + ")") 

            # Save the best model
            if test_makespan < makespan_best:
                makespan_best = test_makespan
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)       
                best_id = id-1
                best_actor = copy.deepcopy(actor)
                save_file = '{0}/save_best_{1}_{2}_gen({3})_id({4})_r({5}).pt'.format(save_path_ne, num_jobs, num_mas, i, id, round(test_makespan.item(),2))
                save_file_2 = '{0}/save_best_{1}_{2}.pt'.format(model_path_ne, num_jobs, num_mas)
                
                best_models.append(save_file)

                save_model = model.policy.state_dict()
                torch.save(save_model, save_file)
                torch.save(save_model, save_file_2)

        print("")
        print("Fitness    (Avg:", '%.3f' % np.mean(all_fitness), "| Max:", '%.3f' % np.max(all_fitness), "| Min:", '%.3f' % np.min(all_fitness) + ")")
        print("Makespan   (Avg:", '%.3f' % np.mean(all_makespans), "| Max:", '%.3f' % np.max(all_makespans), "| Min:", '%.3f' % np.min(all_makespans) + ")")
        print("Validation (Avg:", '%.3f' % np.mean(test_makespans), "| Max:", '%.3f' % np.max(test_makespans), "| Min:", '%.3f' % np.min(test_makespans) + ")")

        train_time = time.time()-last_time
        print("train_time: {:.3f}s".format(train_time))

        # Save the best result
        result_makespans.append(np.min(test_makespans))



        # DRL training
        if i % train_paras["sync_iter"] == 0:
            ids = find_ids(pop) #Find top 3 most different individuals

            data = [env_paras, model_paras, train_paras, env_valid_paras, env_valid, best_critic,
                    copy.deepcopy(drl_best_models), copy.deepcopy(drl_best_models_2), drl_makespan_best, 
                    save_path, save_path_nn, model_path_nn, last_time, i]

            T1 = Thread(1, ids[0], pop[ids[0]], data)
            T2 = Thread(2, ids[1], pop[ids[1]], data)
            T3 = Thread(3, ids[2], pop[ids[2]], data)

            T1.start(); T2.start(); T3.start()
            T1.join();  T2.join();  T3.join()

            pop[ids[0]] = T1.drl_actor
            pop[ids[1]] = T2.drl_actor
            pop[ids[2]] = T3.drl_actor
            best_critic = T1.drl_critic
            
            max_fitness = np.max(all_fitness)
            all_fitness[ids[0]] = max_fitness
            all_fitness[ids[1]] = max_fitness
            all_fitness[ids[2]] = max_fitness

            drl_makespan_best = [T1.mk_best, T2.mk_best, T3.mk_best]


            # ERL 2.0 code...
            # mk_best, drl_actor, drl_critic = drl_training(0, best_id, best_actor, data)
            
            # best_critic = drl_critic
            # best_actor = copy.deepcopy(drl_actor)
            # pop[best_id] = drl_actor

            # drl_makespan_best = mk_best

            # all_fitness[best_id] = max_fitness

            # print("\nID({0}): fitness: {1} wgt_sum: {2}\n".format(best_id, best_actor.fitness, best_actor.sum_params()[0]))

        # Evolution
        evolver.epoch(i, pop, all_fitness)

    print("\ntotal_time: {:.3f}s".format(time.time()-start_time))

if __name__ == '__main__':
    main(int(sys.argv[-4]), sys.argv[-3], sys.argv[-2], sys.argv[-1])
