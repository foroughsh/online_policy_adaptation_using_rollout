# Structure of the data traces collected from the testbed

We save the data traces collected from the testbed in a .csv file. 
In this file, each line includes the data of each sample collected in each time step (i.e., aggregated information over a monitoring step that is 5 seconds).
Since we use supervised learning to learn the system model, we can divide the columns of this file into two parts, namely the feature set and targets. The feature set, includes the offered load for the services, the carried load of the system for each service, the allocated CPU for each pod, routing weights for each service towards each processing node, and the blocking rate of each service in the front node. 
