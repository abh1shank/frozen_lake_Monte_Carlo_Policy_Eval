import numpy as np

def mc_state_value_estimation(env,state_num,num_episodes,discount):
    sum_return=np.zeros(state_num)
    num_visits=np.zeros(state_num)
    value_fn_estimate=np.zeros(state_num)

    for i in range(num_episodes):
        visited_states=[]
        reward_gain=[]
        (cur_state,prob)=env.reset()
        visited_states.append(cur_state)
        print("episode number:",i+1)
        while 1:
            random_action=env.action_space.sample()
            (state,reward,terminal,_,_)=env.step(random_action)
            reward_gain.append(reward)
            if terminal: break
            else:
                visited_states.append(state)
    num_visited_States=len(visited_states)
    #gt=rt+ y rt+1 +y^2 rt+2+ ....
    gt=0
    for i in range(num_visited_States-1,-1,-1):
        state_temp=visited_states[i]
        return_temp=reward_gain[i]
        gt=discount*gt+return_temp

        if state_temp not in visited_states[0:i]:
            num_visits[state_temp]+=1
            sum_return[state_temp]+=gt
    for i in range(state_num):
        if num_visits[i]!=0:
            value_fn_estimate[i]=sum_return[i]/num_visits[i]

    return value_fn_estimate

def eval_policy(env, V, policy, gamma, max_iter, tol):
    import numpy as np
    conv_track = []
    for it in range(max_iter):
        conv_track.append(np.linalg.norm(V, 2))
        V_next = np.zeros(env.observation_space.n)
        for s in env.P:
            outer_sum = 0
            for a in env.P[s]:
                inner_sum = 0
                for p, next_s, r, is_terminal in env.P[s][a]:
                    inner_sum += p * (r + gamma * V[next_s])
                outer_sum += policy[s, a] * inner_sum
            V_next[s] = outer_sum
        if np.max(np.abs(V_next - V)) < tol:
            V = V_next
            print('Iterative policy evaluation algorithm converged!')
            break
        V = V_next
    return V

def grid_print(V, reshape_dim, filename):
    import seaborn as sns
    import matplotlib.pyplot as plt
    ax = sns.heatmap(V.reshape(reshape_dim, reshape_dim),
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    plt.savefig(filename, dpi=600)
    plt.show()
