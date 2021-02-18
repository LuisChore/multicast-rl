import numpy as np
import math
from multiAgentEnv import MultiCast
from Graph import Graph


episodes = 10
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

def main():
    f = open("example.txt", "r")
    nodes,edges = f.readline().split()
    g = Graph(int(nodes),True)
    Q = {}
    Qc = {}
    for i in range(int(edges)):
        u,v,w = f.readline().split()
        g.add_edge(int(v),int(u),int(w)) #reversed, from agents to target
        Q[(int(v),int(u))] = inf
        Qc[(int(v),int(u))] = 0
    agents = [int(x) for x in f.readline().split()]
    target = int(f.readline())
    env = MultiCast(g,agents,target)
    for episode in range(episodes):
        states_actions_returns = play_iteration(env,Q)
        for s,a,G in states_actions_returns:
            for si,ai in zip(s,a):
                if (si,ai) in Q:
                    print("->" + str((si,ai)))
                    if Qc[(si,ai)] == 0:
                        Q[(si,ai)] = G
                        Qc[(si,ai)] = 1
                    else:
                        qc = Qc[(si,ai)]
                        Q[(si,ai)] = (Q[(si,ai)]*qc + G) / (qc + 1)
                        Qc[(si,ai)] += 1

        for u in range(0,g.nodes):
            for v,_ in g.adj[u]:
                print( str((u,v)) + " : " + str(Q[(u,v)]))
    eps = 0
    G = play_iteration(env,Q,True,True)
    print("Solution: " + str(G))
if __name__ == "__main__":
    main()
