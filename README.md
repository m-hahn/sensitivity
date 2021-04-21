# Code

## Sensitivity Bounds for ML Models (Section 3)

* randomly initialized LSTMs

[Results](code/learnability/output/lstm-init-s1.pdf)

* LSTM learnability 

[Results](code/learnability/output/learnability3_together.pdf)

* Transformer learnability (done after the paper was finished):

## Sensitivity of NLP Tasks (Section 4)

* generating alternatives using XLNet

* generating alternatives using u-PMLM: see https://github.com/m-hahn/PMLM

* creating RoBERTa predictions: see https://github.com/m-hahn/fairseq

* creating BoE and LSTM predictions

* creating GPT-2 predictions for Syntax tasks

* results across tasks for XLNet

[Results: GLUE](code/analyze/joint_GLUE.pdf)
[Results: Parsing](code/analyze/joint_Parsing.pdf)
[Results: Syntax](code/analyze/joint_Syntax.pdf)
[Results: Text Classification](code/analyze/joint_textclas.pdf)

* results across tasks for u-PMLM (not in paper due to space limitations)

* Sensitivity and length

[Results: Text Classification and CoLA](code/analyze/byLength_s1ensitivity_textclas_cola.pdf)
[Results: By Task Group](code/analyze/byLength_s1ensitivity_textclas_glue.pdf)

* performance of BoE, LSTN, RoBERTa

XLNet:

[Results](code/analyze/s1ensitivity-accuracy-grid.pdf)
        
    u-PMLM:
    
[Results](code/analyze/s1ensitivity-accuracy-grid-pmlm.pdf)

* per-input analysis

[Results: Sensitivity and Label Dispersion](code/perExample/outputs/subspans_s1ensitivity_rev.pdf)
[Results: Sensitivity and Accuracy](code/perExample/outputs/s1ensitivity_accuracy_roberta-cbow-lstm.pdf)

* Role of task model

* Evaluating lower-bound approximation

## Human Studies (Section 5)

* Experiment 1
** Code:
** Link to online experiment:
** Results:

[Results](experiments/100-rte/Submiterator-master/figures/rte_sst_sensitivities_expt1.pdf)

* Experiment 2
** Code:
** Link to online experiment:
** Results:
[Results](experiments/102-rte/Submiterator-master/figures/sensitivity-changes-sst2-rte.pdf)

