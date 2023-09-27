# Structure of the data traces collected from the testbed

We save the data traces collected from the testbed in a .csv file. 
In this file, each line includes the data of each sample collected in each time step (i.e., aggregated information over a monitoring step that is 5 seconds).
Since we use supervised learning to learn the system model, we can divide the columns of this file into two parts, namely the feature set and targets. The feature set, includes the offered load for the services, the carried load of the system for each service, the allocated CPU for each pod, routing weights for each service towards each processing node, and the blocking rate of each service in the front node. The following figure shows the structure of the service mesh and components and the configuration controls that we apply in our testbed. 

<p align="center">
<img src="https://github.com/foroughsh/online_policy_adaptation_using_rollout/blob/main/documentation/usecase_graph%20(3).png" width="350"/>
</p>

The following figure shows the example file with only one service. All the metrics collected in this file are from the application layer, however, we can collect features from the OS layer and orchestration layer too. In this file the values in the red box are the feature set and the values in the black box are the QoS metrics related to the response time of service 1.

<p align="center">
<img src="https://github.com/foroughsh/online_policy_adaptation_using_rollout/blob/main/documentation/data_structure.png" width="700"/>
</p>
