# Continual Learning with BERT for Questions and Answers
This is the final project for GA-DS 1012 Natural Language Understanding course at NYU. We plan to implement continual learning with BERT over questions and answers tasks using Elastic Weights Consolidation (EWC).

## The Baseline model
For the baseline part, we first used BERT model to train on SQuAD v2.0 (Task A), and got F1 score after evaluation, and then we let the model learn over NewsQA (Task B),  and evaluated back on the SQuAD (Task A) without further fine-tuning the model.

## EWC implementation
There are three main functions we involved and modified and we put it in this folder. And data should be downloaded separately. To see the original code we wrote, see [here](https://github.com/JasonZhangzy1757/mrqa_for_nlu).

We implemented EWC over the BERT model so that the parameters could be regularized during the second learning phases and information that was learned from Task A could be preserved. Furthermore, we also evaluate NewsQA (Task B) to check if this kind of regularization undermines the performance of it since the parameters have been penalized. In order to achieve this, we used a pre-trained BERTBASE model (Devlin et al., 2018) as our guide, we also learned from an unofficial PyTorch implementation of DeepMind's paper (Kirkpatrick, 2017). 

In the experiment, we used default BERT hyperparameters from pytorch_pretrained_bert, and we used  λ=40 as EWC regularization parameter. During the training, We set the learning rate to 3e-5, which is the suggested learning rate for using BERT, according to preliminary experiments. We used a training batch size of 8 and used 6 GPU and 64GB of memory. We chose a relatively small batch size to prevent cuda from being overloaded and out of memory. And we used 2 epochs because in general cases this will give a reasonably acceptable score.


### EWC demo 
EWC demo is an implementation of Elastic Weight Consolidation (EWC), proposed in James Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks 2016(10.1073/pnas.1611835114). Trying to understand how the codes works, we studied many versions of EWC implementation over different tasks on bert.

##### *Special thanks to @하준수, @seanie12 whose code we based heavily on.
##### *Special thanks to @yungshun317, @moskomule, @ariseff, whose codes give us a lot of inspirations.
