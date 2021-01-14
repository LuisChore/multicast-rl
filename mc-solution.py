import numpy as np
import math
from multiAgentEnv import MultiCast
from Graph import Graph


episodes = 100000
gamma = 1 # discount factor
eps = 0.5

inf = float('inf')



def print_policy(index,graph,policy):
    for v,_ in graph.adj[index]:
        print(v,policy[(index,v)],end = ",")
    print()

def choose_next(index,graph,policy):
    p = np.random.random()
    rn = np.random.randint(0,len(graph.adj[index]))
    if p < eps:
        return graph.adj[index][rn][0]
    else:
        best = graph.adj[index][rn][0]
        best_value = policy[(index,best)]
        for v,_ in graph.adj[index]:
            if policy[(index,v)] < best_value:
                best = v
                best_value = policy[(index,v)]
        return best

def choose_action(env,policy):
    action = []
    for agent in env.agents:
        if agent == env.target:
            action.append(agent)
        else:
            action.append(choose_next(agent,env.graph,policy))
    return action


# missing validation if it's already done without iteration
def play_iteration(env,policy,render = False, answer = False):
    state = env.reset()
    done = False
    states_actions_costs = []
    action = choose_action(env,policy)
    # state_t, action_t, cost_{t-1}
    s = tuple(state)
    a = tuple(action)
    states_actions_costs.append((s,a,0))
    while done == False:
        if render:
            env.render()
        state,cost,done = env.step(action)
        s = tuple(state)
        if done:
            states_actions_costs.append((s,None,cost))
        else:
            action = choose_action(env,policy)
            a = tuple(action)
            states_actions_costs.append((s,a,cost))
    if render:
        env.render()
    G = 0 # sum of future costs
    #backwards computation
    states_actions_returns = []
    first = True
    for s,a,c in reversed(states_actions_costs):
        if first:
            first = False
        else:
            states_actions_returns.append((s,a,G))
        G = c + gamma * G
    states_actions_returns.reverse()
    if answer:
        return G
    else:
        return states_actions_returns


def modify_policy(PI,s,a,value):
    for si,ai in zip(s,a):
        if (si,ai) in PI:
            PI[(si,ai)] = min(PI[(si,ai)],value)

def main():
    f = open("example.txt", "r")
    nodes,edges = f.readline().split()
    g = Graph(int(nodes),True)
    PI = {}
    for i in range(int(edges)):
        u,v,w = f.readline().split()
        g.add_edge(int(v),int(u),int(w)) #reversed, from agents to target
        PI[(int(v),int(u))] = inf
    agents = [int(x) for x in f.readline().split()]
    target = int(f.readline())
    env = MultiCast(g,agents,target)
    Q = {}
    Qc = {}
    for episode in range(episodes):
        states_actions_returns = play_iteration(env,PI)
        for s,a,G in states_actions_returns:
            if s in Q:
                if a in Q[s]:
                    qc = Qc[s][a]
                    Q[s][a] = (Q[s][a]*qc + G) / (qc + 1)
                    Qc[s][a] += 1
                else:
                    Q[s][a] = G
                    Qc[s][a] = 1
            else:
                Q[s] = {}
                Qc[s] = {}
                Q[s][a] = G
                Qc[s][a] = 1
            modify_policy(PI,s,a,Q[s][a])
    eps = 0
    G = play_iteration(env,PI,True,True)
    print("Solution: " + str(G))
if __name__ == "__main__":
    main()
