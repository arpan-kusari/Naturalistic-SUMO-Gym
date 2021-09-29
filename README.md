# Naturalistic-SUMO-Gym

SumoGym is a python package to help utilize SUMO as a testing and validation platform for autonomous vehicles without the hassle of creating specific scenarios. 

It uses OpenAI Gym to abstract the process of generation and running the simulation environment. There are four specific steps which are inherited from OpenAI gym environment:
    
	env = SumoGym(scenario, choice, delta_t, render_flag)
	obs = env.reset()
	new_obs, reward, done, info = env.step(action)
	env.close()
	
We also provide some sample route files for two specific scenario types: Highway and Urban. For each scenario type, we have net.xml files which are taken from real maps (e.g. the highway network is around Ann Arbor area). The rou.xml files are populated by vehicles with car-following parameters sampled from naturalistic distributions as shown in figure below:
![Highway IDM parameters](/images/Parameter_IDM_highway.png)

The paper accompanying the code is given in [arXiv link](https://arxiv.org/abs/2109.11620).