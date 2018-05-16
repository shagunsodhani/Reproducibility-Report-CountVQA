# Reproducibility-Report-CountVQA

**Note**: Please set the data paths in config.py before running the code. Also create a dir. named 'logs/' in the main code dir.

1. Run 'python train.py' to start training. The checkpoints will be dumped in the logs/ dir.
1. Run 'python dump_results.py --val --ckpt_file <>' to dump the predictions for validation set in JSON format.
1. Run 'python dump_results.py --test --ckpt_file <>' to dump the predictions for test set in JSON format.
1. Run 'python dump_results.py --test_dev --ckpt_file <>' to dump the predictions for test-dev set in JSON format.
1. Run 'python plot.py <ckpt_file_path> full' to plot the garphs corresponding to peicewise linear functions f_1 to f_8.
1. Run 'python plot_metrics.py  --ckpt_file <>' to plot the graphs corresponding to train, validation loss and accuracy.
1. Run 'python eval-acc.py <ckpt_file_path>' for computing the validation set results (also on balanced pairs).