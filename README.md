# Continual Learning with BERT for Questions and Answers
This is the final project for GA-DS 1012 Natural Language Understanding course at NYU. We plan to implement continual learning with BERT over questions and answers tasks using Elastic Weights Consolidation (EWC).

## The Baseline model
For the baseline part, we first used BERT model to train on SQuAD v2.0 (Task A), and got F1 score after evaluation, and then we let the model learn over NewsQA (Task B),  and evaluated back on the SQuAD (Task A) without further fine-tuning the model.

## EWC implementation
There are three main functions we involved and modified and we put it in this folder. And data should be downloaded separately. To see the original code we wrote, see [here](https://github.com/JasonZhangzy1757/mrqa_for_nlu).

We implemented EWC over the BERT model so that the parameters could be regularized during the second learning phases and information that was learned from Task A could be preserved. Furthermore, we also evaluate NewsQA (Task B) to check if this kind of regularization undermines the performance of it since the parameters have been penalized. In order to achieve this, we used a pre-trained BERTBASE model (Devlin et al., 2018) as our guide, we also learned from an unofficial PyTorch implementation of DeepMind's paper (Kirkpatrick, 2017). 

In the experiment, we used default BERT hyperparameters from pytorch_pretrained_bert, and we used  λ=40 as EWC regularization parameter. During the training, We set the learning rate to 3e-5, which is the suggested learning rate for using BERT, according to preliminary experiments. We used a training batch size of 8 and used 6 GPU and 64GB of memory. We chose a relatively small batch size to prevent cuda from being overloaded and out of memory. And we used 2 epochs because in general cases this will give a reasonably acceptable score.

## Result & Conclusion

Results for the experiments are listed in table 1. The baseline shows without any regularization, NewsQA(Task B) gets F1 score 61.72. And based on NewsQA’s parameters, SQuAD(Task A)’s F1 score drops from 84.66 to 73.62 which demonstrates the catastrophic forgetting really exists in the QA domain. Then we implemented EWC on our model, the performance of NewQA reduced by 5.48% with F1 score 58.34. This slight drop is resulted from taking the important parameters in SQuAD into penalty. SQuAD’ s performance increases by 3.0% with F1 score 75.85. This indicates that EWC has a slight regularization effect. It alleviates catastrophic forgetting to some extent, but the improvement is not significant.

In the meanwhile, we compare the severity of catastrophic forgetting between text classification and Q&A.  BERT performs poorly on text classification when a new dataset is added with accuracy near 0. However, the baseline model performs much better in Q&A with a decrease by 13% which still has an F1 score over 70. This indicates BERT itself suffers less catastrophic forgetting in QA domain. 

![img](https://github.com/JasonZhangzy1757/NLU_Final_Project/blob/master/result.png)

To conclude, in this research, we propose a continual learning approach EWC to solve the catastrophic forgetting problem in QA domain. Our research highlights that EWC can reduce catastrophic forgetting. Our research also shows BERT suffers less catastrophic forgetting in QA problems compared with text classification problems. Future work includes adding multiple tasks with different sequences to analyze the influence of sequence of tasks, and using different regularization methods to compare the performance of reducing catastrophic forgetting with EWC.


## Data Preperation
### Download the original data
All the data are in data folder including SQuAD and NewsQA. If you want to get more datasets,please use 'wget link'
to download the datasets
MRQA datasets: https://mrqa.github.io/shared

## Requirements
Please install the following library requirements specified in the **requirements.txt** first.

```bash
torch==1.1.0
pytorch-pretrained-bert>=0.6.2
json-lines>=0.5.0
```

## Model Training & Validation

```bash
$ python3 main.py \
         --epochs 2 \
         --batch_size 64 \
         --lr 3e-5 \
         --do_lower_case \
         --use_cuda \
         --devices 0_1_2_3
```
- If you are using uncased bert model, give the option `--do_lower_case`


### EWC demo 
EWC demo is an implementation of Elastic Weight Consolidation (EWC), proposed in James Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks 2016(10.1073/pnas.1611835114). Trying to understand how the codes works, we studied many versions of EWC implementation over different tasks on bert.

##### *Special thanks to @하준수, @seanie12 whose code we based heavily on.
##### *Special thanks to @yungshun317, @moskomule, @ariseff, whose codes give us a lot of inspirations.
