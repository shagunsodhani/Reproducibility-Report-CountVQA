# Reproducibility-Report-CountVQA

This is the implementation of the ICLR 2018 paper [Learning to Count Objects in Natural Images for Visual Question Answering](https://openreview.net/forum?id=B12Js_yRb) to accompany the reproducibility report for the paper. Check here for the [official implementation](https://github.com/Cyanogenoid/vqa-counting). For instructions on setting up the code in this repository, skip to [setup-instructions](#setup-instructions).

## Summary of the paper

Most of the visual question-answering (VQA) models perform poorly on the task of counting objects in an image. The reasons are manifold:

  * Most VQA models use a soft attention mechanism to perform a weighted sum over the spatial features to obtain a single feature vector. These aggregated features help in most categories of questions but seem to hurt for counting based questions.
  * For the counting questions, we do not have a ground truth segmentation of where the objects to be counted are present on the image. This limits the scope of supervision.

Additionally, we need to ensure that any modification in the architecture, to enhance the performance of the counting questions, should not degrade the performance on other classes of questions.

The paper proposes to overcome these challenges by using the attention maps (and not the aggregated feature vectors) as the input to a separate **count** module. The basic idea is quite intuitive: when we perform weighted averaging based on different attention maps, we end up averaging the features corresponding to the different instances of an object. This makes the feature vectors indistinguishable from the scenario where we had just one instance of the object in the image. Even multiple glimpses (multiple steps of attention) cannot resolve this problem as the weights given to one feature vector would not depend on the other feature vectors (that are attended to). Hard attention could be more useful than soft-attention but there is not much empirical evidence in support of this hypothesis. 


## Summary of the reproducibility report

We implemented both the baseline architecture and the count module from scratch using the details mentioned in the paper and the hyper-parameters suggested by the paper to reproduce the experiments on the VQA-v2 dataset. We also implemented the baseline architecture, which does not use a separate count module. We report the results on test-dev and test, as obtained from the official evaluation server. For the validation set, we report results on individual question categories as well as balanced pairs by using the evaluation script provided by the authors for our trained model. We also report the results on the validation set as obtained from the official evaluation scripts released by \cite{goyal2017making}. In a nutshell, our experiments achieve the similar level of performance gains, compared to the baseline model, as reported in the paper. For a detailed discussion on this aspect and the assumptions we made in our implementation, refer to the [reproducibility report](TBD). 

We note that we could not exactly reproduce the results in the paper, but the more likely reason is the limit in terms of computational resources. Our independent implementation provide empirical evidence of the effectiveness of the proposed model.

In addition to the experiments proposed in the paper, we performed some more experiments to perform a more holistic ablation study on the choice of different hyper-parameters or sub-modules of the model proposed by the authors. The following additional experiments were performed: 
  
  * Increasing the number of objects to 20 instead of default 10.
  * Use of unidirectional LSTM instead of unidirectional GRU to obtain the question features.
  * Setting the threshold value for the confidence of prediction to 0.2 instead of 0.5.
  * Setting the embedding dimension corresponding to the words to 100 instead of 300.
  * Use of second glimpse attention vector instead of first glimpse attention vector.
  * Use of bidirectional LSTM instead of the unidirectional LSTM for the baseline model.
  
Our assessment shows that the model is largely robust to choice of different hyper-parameters and gives significant gains in performance over count-based questions, compared to the baseline model.

## Results



## Setup Instructions

* Set the appropriate data paths in `config.py` before running the code. Also create a directory named `logs/` in the root directory.
* Run`python train.py` to start training. The checkpoints will be dumped in the `logs` directory.
* Run `python dump_results.py --val --ckpt_file <>` to dump the predictions for validation dataset set (in JSON format).
* Run `python dump_results.py --test --ckpt_file <>` to dump the predictions for test dataset (in JSON format).
* Run `python dump_results.py --test_dev --ckpt_file <>` to dump the predictions for test-dev set in JSON format.
* Run `python plot.py <ckpt_file_path> full` to plot the garphs corresponding to piece-wise linear functions `f_1` to `f_8`.
* Run `python plot_metrics.py  --ckpt_file <>` to plot the graphs corresponding to train, validation loss and accuracy.
* Run `python eval-acc.py <ckpt_file_path>` for computing the validation set results (including the results on the balanced pairs).

## References

```
@InProceedings{zhang2018vqacount,
  author    = {Yan Zhang and Jonathon Hare and Adam Pr\"ugel-Bennett},
  title     = {Learning to Count Objects in Natural Images for Visual Question Answering},
  booktitle = {International Conference on Learning Representations},
  year      = {2018},
  eprint    = {1802.05766},
  url       = {https://openreview.net/forum?id=B12Js_yRb},
}
```
