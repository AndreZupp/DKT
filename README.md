# DKT
Distributed Knowledge Transfer (DKT) method for Distributed Continual Learning

Master thesis project for University of Pisa. This work introduces the <b>Distributed Continual Learning</b> (DCL) research area and the
<b>Distributed Knowledge Transfer </b> (DKT) architecture.

<b>Please notice that the code uploaded here needs pre-trained model to work, do not esitate to contact me for further information. </b>


## Distributed Continual Learning 
In Distributed Continual Learning (DCL), the term ”distributed” refers to a complex and highly interconnected environment that involves multiple agents working
together to improve their performance through the exchange of information during
the training process. What distinguishes the DCL approach is fusion with the continual learning environment, whereby models continuously give and take their state
with each other at regular intervals, creating a highly dynamic and adaptive training
process.


## Distributed Knowledge Transfer 
The proposed method applies knowledge distillation to the distributed continual scenario.
The proposed architecture attaches two distinct classification heads (fig. 3.2) to a
feature extractor. The first head, called continual learning head (CL), uses cross-
entropy loss to optimize the model performance on the hard targets of the current
experience, while the second head, called student head (ST), adopts another loss
function (typically KD loss or MSE) using as target the predictions of another
model on the very same experience.

<br/><br/>
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/45973023/236632079-d5bcc389-2dce-4e58-a9ec-70d625a41c50.png">
</p>
<br/><br/>

The loss function is the sum of two head-specific loss functions:
<br/><br/>
```math
\begin{equation}
    \mathcal{L} = \mathcal{L}_{cl} + \mathcal{L}_{kd} 
\end{equation}
```
<br/><br/>

The first term of the sum is relative to the Continual Learning (CL) head. It is the classic cross-entropy between the target $t_i$ and $q_{cl}$,
the soft-max of the cl head
<br/><br/>
```math
\begin{equation}
    \mathcal{L}_{cl} = \mathcal{L}_{ce}(q^{cl}) = - \sum_i^C t_i \log{q_{cl}^{(i)}}
\end{equation}
```
<br/><br/>
The second term is relative to the Student head (ST). It is the KD loss (in this case MSE has been used)
between q&#x302;<sub>tc</sub> the soft-targets of the teacher model, q&#x302;<sub>cl</sub>
the soft-targets of the student head distilled at the same temperature
<br/><br/>
```math
\begin{equation}
    \mathcal{L}_{st} = \mathcal{L}_{kd}(q^{st}) = - \sum_i^n \frac{(\hat{q}_{tc}^{(i)} - \hat{q}_{st}^{(i)})^2}{n}
\end{equation}
```
<br/><br/>

## Requirements 
The requirements are contained in the requirements.txt file and can be installed via pip:

`pip install -r requirements.txt`

The project has been developed using the <a href="https://avalanche.continualai.org/">Avalanche Continual Learning Library</a>, which is based on Pytorch. 
Most of the dependencies are already contained in the Avalanche installation. 

Please notice that the requirements were rawly extracted from the conda environment hence they need to be pruned.

You can use newer version of Cuda/Pytorch as I was limited by the NVIDIA driver of the machine I was working on.

## How to run 

The thesis consisted of three experiments and each one can be run with the specific script

Experiment1 -----> cifar100_training.py
Experiment2 -----> splitcifar100_pretrained.py
Experiment3 -----> step_training.py 

They can be executed via python: 

`python cifar100_training.py`

### Further information
If you want to know more about this project you can consult my <a href="https://etd.adm.unipi.it/theses/available/etd-03092023-161823/">Master Thesis</a>





