The following figure shows the target system that underlies our evaluation scenarios. It is a service mesh consisting of three services provided by five microservice components. 
The service mesh is realized using Kubernetes and Istio and runs on top of a server cluster connected through a Gigabit Ethernet switch. 
The cluster contains nine Dell PowerEdge R715 2U servers, each with $64$ GB RAM, two 12-core AMD Opteron processors, a 500 GB hard disk, and four 1 Gb network interfaces. 
The tenth machine is a Dell PowerEdge R630 2U with 256 GB RAM, two 12-core Intel Xeon E5-2680 processors, two $1.2$ TB hard disks, and twelve 1 Gb network interfaces. 
All machines run Ubuntu Server 18.04.6 64 bits and their clocks are synchronized through NTP.

The service mesh includes three services: two information services ($S_1, S_3$) and one compute service ($S_2$). 
Requests for the two information services include object identifiers and the corresponding responses include information about the objects fetched from a database 
(each information service fetches data from a different database deployed on both database nodes). A request for the compute service causes the service to multiply 
two random matrices of the sizes $2\,000\times1\,000$ and $\,1000\times 2\,000$ and return the result.

<p align="center">
<img src="https://github.com/foroughsh/online_policy_adaptation_using_rollout/blob/main/documentation/images/microservice.png" width="500"/>
</p>
