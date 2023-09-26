# Online Policy Adaptation for Networked Systems using Rollout

Dynamic resource allocation in networked systems is necessary to achieve end-to-end management objectives. Previous research has demonstrated that reinforcement learning is a promising approach to this problem, allowing to obtain near-optimal resource allocation policies for non-trivial system configurations. Despite these advances, a significant drawback of current approaches is that they require expensive and slow retraining whenever the target system changes. We address this drawback and introduce an efficient approach to adapt a given base policy to dynamic system changes. In our approach, we adapt the base policy through rollout and online play, which transforms the base policy into a rollout policy. 

The following figure shows our approach for policy adaptation in networked systems. During each control cycle, the system model $f$ is estimated from system metrics using supervised learning; a given base policy $\hat{\pi}$ is adapted for the current state and the current system model through one step of policy iteration, which we call rollout and the output of this step is an improved rollout policy $\tilde{\pi}$ which is used to select the next control action.

<p align="center">
<img src="https://github.com/foroughsh/OnlinePolicyAdaptationUsingRollout/blob/main/online_policy_adaptation.png" width="500"/>
</p>
