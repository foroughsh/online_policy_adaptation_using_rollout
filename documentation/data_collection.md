# Data collection on the testbed to learn the system model

One of the important steps in our framework is the learning model. The system model is a function that maps the current state of the system and the control action to the next state.
$$s_{t+1} \triangleq f(s_t, a_t, w_t, t) \ \ t=1,2,... $$

where $f$ is a Markovian system model, $w_t \in \mathcal{W}$ is a random disturbance, and $a_t$ is the control action at time $t$, which is defined as

```math
a_t \triangleq ((a^{(p)}_{(j,k),i,t}, a^{(b)}_{i,t}, a^{(c)}_{j,t}))_{i \in \mathscr{S}, (j,k) \in \mathcal{E}, k,j \in \mathcal{V}} math```

where $$a^{(p)}_{(j,k),i,t} \in \{-\Delta_p, 0, \Delta_p\}$$ indicates the change in routing weight for edge $(j,k)$ and service $S_i$, $a^{(b)}_{i,t} \in \{-\Delta_b, 0, \Delta_b\}$ indicates the change in blocking rate for service $S_i$, and $a^{(c)}_{j,t} \in \{-\Delta_c,0,\Delta_c\}$ indicates the change in allocated CPU cores for node $j$.

Given (\ref{eq:action_def}), the system model (\ref{eq:dynamics_def}) can be stated more explicitly as

```math
w_{t+1} &\sim P(\cdot \mid s_t, a_t) && \\
l_{i,t+1} &= \lambda_i(t+1,w_{t+1})&& i \in \mathscr{S} \\
d_{i,t+1} &= \alpha_i(s_{t},a_t, w_{t+1})&& i \in \mathscr{S} \\
p_{(j,k),i,t+1} &= p_{(j,k),i,t} + a^{(p)}_{(j,k),i,t} && i \in \mathscr{S}, (j,k) \in \mathcal{E}\\
b_{i,t+1} &= b_{i,t} + a^{(p)}_{i,t}  && i \in \mathscr{S}\\
c_{j,t+1} &= c_{j,t} + a^{(c)}_{j,t} && k \in \mathcal{V}
```

where $t=1,2,\hdots$, $w_{t+1} \sim P(\cdot \mid s_t, a_t)$ denotes that $w_{t+1}$ is sampled from $P$, and $\alpha_i$ is a function that models the response time of service $S_i$.
