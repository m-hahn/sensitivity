# Code

## Sensitivity Bounds for ML Models (Section 3)

* randomly initialized LSTMs

[Results](code/learnability/output/lstm-init-s1.pdf)

* LSTM learnability 

[code/learnability/output/learnability3_together.pdf](Results)

* Transformer learnability (done after the paper was finished):

## Sensitivity of NLP Tasks (Section 4)

* generating alternatives using XLNet

* generating alternatives using u-PMLM: see https://github.com/m-hahn/PMLM

* creating RoBERTa predictions: see https://github.com/m-hahn/fairseq

* creating BoE and LSTM predictions

* creating GPT-2 predictions for Syntax tasks

* results across tasks for XLNet

[code/analyze/joint_GLUE.pdf](Results: GLUE)
[code/analyze/joint_Parsing.pdf](Results: Parsing)
[code/analyze/joint_Syntax.pdf](Results: Syntax)
[code/analyze/joint_textclas.pdf](Results: text Classification)

* results across tasks for u-PMLM (not in paper due to space limitations)

* Sensitivity and length

[code/analyze/byLength_s1ensitivity_textclas_cola.pdf](Results: Text Classification and CoLA)
[code/analyze/byLength_s1ensitivity_textclas_glue.pdf](Results: By Task Group)

* performance of BoE, LSTN, RoBERTa

XLNet:

[code/analyze/s1ensitivity-accuracy-grid.pdf](Results)
        
    u-PMLM:
    
[code/analyze/s1ensitivity-accuracy-grid-pmlm.pdf](Results)

* per-input analysis

[code/perExample/outputs/subspans_s1ensitivity_rev.pdf](Results: Sensitivity and Label Dispersion)
[code/perExample/outputs/s1ensitivity_accuracy_roberta-cbow-lstm.pdf](Results: Sensitivity and Accuracy)

* Role of task model

* Evaluating lower-bound approximation

## Human Studies (Section 5)

* Experiment 1
** Code:
** Link to online experiment:
** Results:

[experiments/100-rte/Submiterator-master/figures/rte_sst_sensitivities_expt1.pdf](Results)

* Experiment 2
** Code:
** Link to online experiment:
** Results:
[experiments/102-rte/Submiterator-master/figures/sensitivity-changes-sst2-rte.pdf](Results)

