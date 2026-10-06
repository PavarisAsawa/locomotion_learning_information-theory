from .ES_agent import *

class IntegrateAgent():
    """
    Train Plasticity/Feedforward network with built-in cpg signal
    """
    def __init__(self,agent_cfg):
        self.POPSIZE             = agent_cfg["ES_params"]["POPSIZE"]
        self.RANK_FITNESS        = agent_cfg["ES_params"]["rank_fitness"]
        self.ANTITHETIC          = agent_cfg["ES_params"]["antithetic"]
        self.LEARNING_RATE       = agent_cfg["ES_params"]["learning_rate"]
        self.LEARNING_RATE_DECAY = agent_cfg["ES_params"]["learning_rate_decay"]
        self.SIGMA_INIT          = agent_cfg["ES_params"]["sigma_init"]
        self.SIGMA_DECAY         = agent_cfg["ES_params"]["sigma_decay"]
        self.LEARNING_RATE_LIMIT = agent_cfg["ES_params"]["learning_rate_limit"]
        self.SIGMA_LIMIT         = agent_cfg["ES_params"]["sigma_limit"]

        # Models
        self.ARCHITECTURE_NAME   = agent_cfg["model"]
        self.ARCHITECTURE_TYPE   = agent_cfg["model_type"]
        self.FF_ARCHITECTURE     = agent_cfg["FF_ARCHITECTURE"]
        self.HEBB_ARCHITECTURE   = agent_cfg["HEBB_ARCHITECTURE"]
        self.LSTM_ARCHITECTURE   = agent_cfg["LSTM_ARCHITECTURE"]
        self.RBF_ARCHITECTURE    = agent_cfg["RBF_ARCHITECTURE"]

        # ES
        self.HEBB_init_wnoise    = agent_cfg["HEBB_init_wnoise"]
        self.HEBB_norm           = agent_cfg["HEBB_norm"]
        self.USE_TRAIN_HEBB      = agent_cfg["USE_TRAIN_HEBB"]
        self.HEBB_std_init_param      = agent_cfg["HEBB_std_init_param"]

        
        # Training parameters
        self.EPOCHS                  = agent_cfg["EPOCHS"]
        self.EPISODE_LENGTH_TRAIN    = agent_cfg["EPISODE_LENGTH_TRAIN"]
        self.EPISODE_LENGTH_TEST     = agent_cfg["EPISODE_LENGTH_TEST"]
        self.SAVE_EVERY              = agent_cfg["SAVE_EVERY"]
        self.USE_TRAIN_PARAM         = agent_cfg["USE_TRAIN_PARAM"]

        # General Information
        self.TASK              = agent_cfg["task_name"]
        self.TEST              = agent_cfg["test"]
        if self.TEST:
            self.USE_TRAIN_PARAM = True 
        
        # hebb/ff experiment
        self.experiment     = agent_cfg["experiment"] # Name of experiemtn
        # RBF path
        self.rbf_experiment = agent_cfg["rbf_experiment"]

        self.train_ff_path = agent_cfg["train_ff_path"]
        self.train_hebb_path = agent_cfg["train_hebb_path"]
        self.train_lstm_path = agent_cfg["train_lstm_path"]
        self.train_rbf_path = agent_cfg["train_rbf_path"]

        self.device = agent_cfg["rl_device"]

        # Debug WanDB 
        self.wandb_activate = agent_cfg["wandb"]["wandb_activate"]
        self.wandb_name = agent_cfg["wandb"]["wandb_name"]
        self.wandb_group = agent_cfg["wandb"]["wandb_group"]
        self.wandb_project = agent_cfg["wandb"]["wandb_project"]

        if self.wandb_activate:
            # run_name = f"{self.wandb_name}_{self.ARCHITECTURE_NAME}_{self.wandb_group}"
            run_name = f"{self.experiment}"
            wandb.init(
                project=self.wandb_project,
                group=self.wandb_group,
                config=agent_cfg,
                name=run_name,
            )

        # CREATE model ###################################################################

        if self.ARCHITECTURE_NAME == 'ff':
            self.models = FeedForwardNet(popsize=self.POPSIZE,
                                    sizes=self.FF_ARCHITECTURE,
                                    )
            self.dir_path = 'logs/'+'es/'+self.TASK+'/cpg_rbf_ff/'+self.experiment
        elif self.ARCHITECTURE_NAME == 'hebb':
            self.models = HebbianNet(popsize=self.POPSIZE, 
                                sizes=self.HEBB_ARCHITECTURE,
                                init_noise=self.HEBB_init_wnoise,
                                norm_mode=self.HEBB_norm,
                                std_init_param=self.HEBB_std_init_param
                                )
            self.dir_path = 'logs/'+'es/'+self.TASK+'/cpg_rbf_hebb/'+self.experiment
        elif self.ARCHITECTURE_NAME == 'lstm':
            self.models = LSTMs(popsize=self.POPSIZE, 
                        arch=self.LSTM_ARCHITECTURE,
                        )
            self.dir_path = 'logs/'+'es/'+self.TASK+'/cpg_rbf_lstm/'+self.experiment
            raise ValueError(f"Unsupported architecture: {self.ARCHITECTURE_NAME}. Only 'hebb', 'ff', 'LSTM' is supported.")

        # Load model for playing
        if self.USE_TRAIN_PARAM:
            # if testing load the trained parameter
            if self.ARCHITECTURE_NAME == 'ff':
                model_checkpoint = self.train_ff_path
            if self.ARCHITECTURE_NAME == 'hebb':
                model_checkpoint = self.train_hebb_path
            if self.ARCHITECTURE_NAME == 'lstm':
                model_checkpoint = self.train_lstm_path
            trained_data = pickle.load(open(self.dir_path+"/model/"+model_checkpoint, 'rb'))
            
            self.train_params = trained_data[0].best_param()
            self.solver = trained_data[0]
            print('train_params number: ', len(self.train_params))

        ####################################################################
        ###################### ALWAYS LOAD CPG AGENT #######################
        # Load CPG agent
        self.cpg = RBFNet(popsize=self.POPSIZE, 
                                num_basis=self.RBF_ARCHITECTURE[0],
                                num_output=self.RBF_ARCHITECTURE[1],
                                motor_encode='direct',
                                )

        if self.USE_TRAIN_PARAM: # IF TESTING -> keep rbf net. and hebb in same pickle after training 
            cpg_trained_data = pickle.load(open(self.dir_path+"/model/" + model_checkpoint, 'rb'))
            print(type(cpg_trained_data[2]))
            self.cpg_train_params = cpg_trained_data[2].get_a_model_params()
        else: # IF TRAINING
            self.cpg_dir_path = 'logs/'+'es/'+self.TASK+'/cpg_rbf/'+ self.rbf_experiment
            cpg_trained_data = pickle.load(open(self.cpg_dir_path+"/model/"+self.train_rbf_path, 'rb'))
            self.cpg_train_params = cpg_trained_data[0].best_param()
        # self.cpg_solver = cpg_trained_data[0]
        self.cpg.set_a_model_params(self.cpg_train_params)


        ####################################################################
        self.n_params_a_model = self.models.get_n_params_a_model()
        # Set optimizer
        if not self.TEST:
            self.solver = OpenES(self.n_params_a_model,
                    popsize=self.POPSIZE,
                    rank_fitness=self.RANK_FITNESS,
                    antithetic=self.ANTITHETIC,
                    learning_rate=self.LEARNING_RATE,
                    learning_rate_decay=self.LEARNING_RATE_DECAY,
                    sigma_init=self.SIGMA_INIT,
                    sigma_decay=self.SIGMA_DECAY,
                    learning_rate_limit=self.LEARNING_RATE_LIMIT,
                    sigma_limit=self.SIGMA_LIMIT)
            self.solver.set_mu(self.models.get_a_model_params())
            pass

        self.low_pass = True

    def run(self,env, test=False):
        """
        Run the ES agent on the environment.
        :param env: The environment to run the agent on.
        :param train: Whether to train the agent or not.
        """        
        # ES code
        # Log data initialized

        if test:   # Trainig Loop
            self.run_play(env=env)
        else:       # Playing Loop
            self.run_train(env=env)
        
 
    def run_train(self,env):
        pop_mean_curve = np.zeros(self.EPOCHS)
        best_sol_curve = np.zeros(self.EPOCHS)
        eval_curve = np.zeros(self.EPOCHS)

        log = { "reward": [] }                 # episode-wise means
        wandb_log_buffer = {}                  # temp dict for wandb each epoch


        for epoch in tqdm(range(self.EPOCHS)):
            current_ep = epoch + 1

            # sample params from ES and set model params
            solutions = self.solver.ask()
            self.models.set_models_params(solutions)
            
            total_rewards = torch.zeros(self.POPSIZE, device=self.device)
            cumulative_reward = torch.zeros(self.POPSIZE , device=self.device)

            running_totals = {}
            
            obs , _ = env.reset()
            prev_action = None

            # Rollout
            for time_step in range(self.EPISODE_LENGTH_TRAIN):
                cpg_signal = self.cpg.forward(obs["policy"])
                actions = self.models.forward(obs["policy"])
                if prev_action is None:
                    prev_action = torch.zeros_like(actions)
                if self.low_pass:
                    actions = 0.1*actions + 0.9*prev_action
                    prev_action = actions.clone()
                
                next_obs, reward, terminated, truncated, extras = env.step(actions+cpg_signal)
                
                cumulative_reward += reward
                done = torch.logical_or(terminated, truncated)
                obs = next_obs

                # Set Objective Function to ES
                total_rewards += reward/self.EPISODE_LENGTH_TRAIN*100

                # Logging
                for key, value in extras["log"].items():          # value is a tensor
                    if key not in running_totals:                 # first time we see key
                        running_totals[key] = torch.zeros_like(value)
                    running_totals[key] += value


            # Update to ES
            total_rewards_cpu = total_rewards.cpu().numpy()
            fitlist = list(total_rewards_cpu)
            self.solver.tell(fitlist)
            fit_arr = np.array(fitlist)

            # convert running_totals → episode means # Logging all extras value
            episode_means = {k: (v / self.EPISODE_LENGTH_TRAIN).cpu().numpy().mean() for k, v in running_totals.items()}
            # episode_means = {k: (v).cpu().numpy().mean() for k, v in running_totals.items()}

            # add episode_means value to log buffe
            for k, v in episode_means.items():
                log.setdefault(k, []).append(v) 
            
            # print('epoch', epoch, 'mean', fit_arr.mean(dtype=np.float64), 
            print('epoch', epoch, 'mean', np.nanmean(fit_arr), 
                  'best', fit_arr.max(), )
            pop_mean_curve[epoch] = fit_arr.mean()
            best_sol_curve[epoch] = fit_arr.max()
            
            print(f"Now time : {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
                

            # WanDB Log data -------------------------------
            if self.wandb_activate:
                wandb_log_buffer.clear()
                wandb_log_buffer["epoch"]  = epoch
                wandb_log_buffer["mean"]   = float(fit_arr.mean())
                wandb_log_buffer["best"]   = float(fit_arr.max())
                wandb_log_buffer["worst"]  = float(fit_arr.min())
                wandb_log_buffer["std"]    = float(fit_arr.std())
                wandb_log_buffer["reward"] = float(cumulative_reward.cpu().numpy().mean())
                wandb_log_buffer.update({k: float(v) for k, v in episode_means.items()})

                wandb.log(wandb_log_buffer)

            # Save model params and OpenES params
            if (epoch + 1) % self.SAVE_EVERY == 0:
                print('saving..')
                # Create Folder
                save_path = os.path.join(self.dir_path, "model",f"model_{epoch}.pickle" )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Dump file
                pickle.dump((
                    self.solver,
                    copy.deepcopy(self.models),
                    copy.deepcopy(self.cpg),
                    pop_mean_curve,
                    best_sol_curve,
                    ), open(save_path, 'wb'))
        env.close()
        if self.wandb_activate:
            wandb.finish()
        
    
    def run_play(self,env):
    
        for epoch in tqdm(range(self.EPOCHS)):
            # sample params from ES and set model params
            self.models.set_a_model_params(self.train_params)
            
            obs , _ = env.reset()
            prev_action = None
            # Rollout
            for time_step in range(self.EPISODE_LENGTH_TRAIN):
                
                cpg_signal = self.cpg.forward(obs['policy'])
                actions = self.models.forward(obs['policy'])
                
                if prev_action is None:
                    prev_action = torch.zeros_like(actions)
                if self.low_pass:
                    actions = 0.1*actions + 0.9*prev_action
                    prev_action = actions.clone()
                    
                next_obs, reward, terminated, truncated, _ = env.step(cpg_signal+actions)
                obs = next_obs
                
        env.close()
        if self.wandb_activate:
            wandb.finish()