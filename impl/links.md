Distributed Energy-Efficient Multi-UAV Navigation for Long-Term Communication Coverage by Deep Reinforcement Learning : Chi Harold Liu , Senior Member, IEEE, Xiaoxin Ma, Xudong Gao, and Jian Tang , Fellow, IEEE

https://github.com/BIT-MCS/DRL-EC3

Energy-Efficient UAV Control for Effective and Fair Communication Coverage: A Deep Reinforcement Learning Approach : Chi Harold Liu , Senior Member, IEEE, Zheyu Chen, Jian Tang , Senior Member, IEEE, Jie Xu, and Chengzhe Piao

**More info related to above** :

Liu et al. proposed a coverage method to have a system of UAVs cover an area and provide communication connectivity while maintaining energy efficiency and fairness of coverage. The authors utilize an actor–critic-based DDPG algorithm. Simulation experiments were carried out with up to 10 UAVs. For a similar communication coverage application, Liu et al. proposed that the UAVs have their own actor-critic networks for a fully-distributed control framework to maximize temporal mean coverage reward.

Due to the unlimited action space for
the coverage and energy control problem in the UAV network, a deep deterministic policy
gradient (DDPG) method can be used. In this research,
the control problem is complex since it needs to optimize four objectives at the same time:
coverage ability, energy consumption, connectivity, and fairness. Therefore, the DDPG is a
promising solution, and it can be used along with the designed utility of the game model
to achieve more coverage, less energy consumption, and high fairness, while keeping the
UAVs connected all the time. It can also deal with complex state spaces and with
time-varying environments, and it uses powerful deep neural networks (DNNs) to assist
the UAV in making decisions and providing high-quality services for the UAV network.
Moreover, the DDPG has the ability to deal with unknown environments and emergency
scenarios, and it enhances the robustness and reduces the calculation cost of the UAVs.

distributed drl-ec3 : ![](image-4.png)

![](image.png)

![](image-1.png)

![](image-2.png)

![](image-3.png)