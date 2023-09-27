<h1>Overview</h1>

The structure of this framework is shown in the Figure below.

<p align="center">
<img src="https://github.com/foroughsh/online_policy_adaptation_using_rollout/blob/main/documentation/images/framework.png" width="500"/>
</p>

As is shown in this figure, to learn the effective policy on this framework efficiently, we take the following 6 steps:

(1) First we define the RL scenario based on the management objectives, available controls in the system, and state of the system.

(2) We run this scenario on the testbed and monitor the system and collect the monitoring data. 

(3) We use this monitoring data and learn the system model.

(4) We use this system model and set up the simulation. We use this simulation to learn the control policy using an RL agent.

(5) We evaluate the learned policy on the simulator for the seen and unseen load patterns.

(6) We evaluate the learned policy on the testbed for the seen and unseen load pattern.

<h1>Set up the testbed</h1>

To set up the testbed, we need to take the following steps:

(1) Install Kubernetes cluster and Istio as our orchestration tool and the service mesh framework. 

(2) Deploy the application microservices by running the ymal files on the master node of our K8 cluster. We need to also configure the virtual services for Istio. 

(3) Test the deployed application by running one of the clients or by sending a request in the web browser. 

(4) Now the testbed is ready!

For more details please check https://github.com/foroughsh/Framework-for-dynamically-meeting-performanc-objectives
