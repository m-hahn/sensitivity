# Code and Results

## Sensitivity Bounds for ML Models (Section 3)

* randomly initialized LSTMs: [Results](code/learnability/output/lstm-init-s1.pdf)

* LSTM learnability: [Results](code/learnability/output/learnability3_together.pdf)

* Transformer learnability (done after the paper was finished): [Results](code/learnability/output/learnability3_together_Transformer.pdf)

## Sensitivity of NLP Tasks (Section 4)

* generating alternatives: [Code](code/xlnet)

* creating RoBERTa predictions: see https://github.com/m-hahn/fairseq

* results across tasks for XLNet

[Results: GLUE](code/analyze/joint_GLUE.pdf)
[Results: Parsing](code/analyze/joint_Parsing.pdf)
[Results: Syntax](code/analyze/joint_Syntax.pdf)
[Results: Text Classification](code/analyze/joint_textclas.pdf)

* results across tasks for u-PMLM (not in paper due to space limitations)

* Sensitivity and length

[Results: Text Classification and CoLA](code/analyze/byLength_s1ensitivity_textclas_cola.pdf)
[Results: By Task Group](code/analyze/byLength_s1ensitivity_textclas_glue.pdf)

* performance of BoE, LSTN, RoBERTa: [XLNet](code/analyze/s1ensitivity-accuracy-grid.pdf) and       
[u-PMLM](code/analyze/s1ensitivity-accuracy-grid-pmlm.pdf)

* per-input analysis: [Sensitivity and Label Dispersion](code/perExample/outputs/subspans_s1ensitivity_rev.pdf) and [Sensitivity and Accuracy](code/perExample/outputs/s1ensitivity_accuracy_roberta-cbow-lstm.pdf)

## Human Studies (Section 5)

### Experiment 1

Code: [RTE](experiments/100-rte), [SST-2](experiments/200-sst2), [SST-2](experiments/200b-sst2)

Online Experiments: [RTE](https://stanford.edu/~mhahn2/experiments/Robustness-Low-Synergy-and-Cheap-Computation/experiments/100-rte/order-preference.html), [SST-2](https://stanford.edu/~mhahn2/experiments/Robustness-Low-Synergy-and-Cheap-Computation/experiments/200-sst2/order-preference.html), [SST-2](https://stanford.edu/~mhahn2/experiments/Robustness-Low-Synergy-and-Cheap-Computation/experiments/200b-sst2/order-preference.html)

[Results](experiments/100-rte/Submiterator-master/figures/rte_sst_sensitivities_expt1.pdf)

### Experiment 2
Code: [RTE](experiments/102-rte), [SST-2](experiments/202-sst2)

Online Experiments: [RTE](https://stanford.edu/~mhahn2/experiments/Robustness-Low-Synergy-and-Cheap-Computation/experiments/102-rte/order-preference.html), [SST-2](https://stanford.edu/~mhahn2/experiments/Robustness-Low-Synergy-and-Cheap-Computation/experiments/202-sst2/order-preference.html)

[Results](experiments/102-rte/Submiterator-master/figures/sensitivity-changes-sst2-rte.pdf)

