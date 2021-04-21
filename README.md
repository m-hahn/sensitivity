# Code

## Sensitivity Bounds for ML Models (Section 3)

* randomly initialized LSTMs

[code/learnability/output/lstm-init-s1.pdf]

* LSTM learnability 

[code/learnability/output/learnability3_together.pdf]

* Transformer learnability (done after the paper was finished):

## Sensitivity of NLP Tasks (Section 4)

* generating alternatives using XLNet

* generating alternatives using u-PMLM: see https://github.com/m-hahn/PMLM

* creating RoBERTa predictions: see https://github.com/m-hahn/fairseq

* creating BoE and LSTM predictions

* creating GPT-2 predictions for Syntax tasks

* results across tasks for XLNet

[code/analyze/joint_GLUE.pdf]
[code/analyze/joint_Parsing.pdf]
[code/analyze/joint_Syntax.pdf]
[code/analyze/joint_textclas.pdf]

* results across tasks for u-PMLM (not in paper due to space limitations)

* Sensitivity and length

    \includegraphics[width=0.23\textwidth]{code/analyze/byLength_s1ensitivity_textclas_cola.pdf}
    \includegraphics[width=0.23\textwidth]{code/analyze/byLength_s1ensitivity_textclas_glue.pdf}

* performance of BoE, LSTN, RoBERTa

XLNet:

    [code/analyze/s1ensitivity-accuracy-grid.pdf]
        
    u-PMLM:
    
    [code/analyze/s1ensitivity-accuracy-grid-pmlm.pdf]

* per-input analysis

[code/perExample/outputs/subspans_s1ensitivity_rev.pdf]
[code/perExample/outputs/s1ensitivity_accuracy_roberta-cbow-lstm.pdf]

* Role of task model

* Evaluating lower-bound approximation

## Human Studies (Section 5)

* Experiment 1
** Code:
** Link to online experiment:
** Results:

[experiments/100-rte/Submiterator-master/figures/rte_sst_sensitivities_expt1.pdf]

* Experiment 2
** Code:
** Link to online experiment:
** Results:
[experiments/102-rte/Submiterator-master/figures/sensitivity-changes-sst2-rte.pdf]

