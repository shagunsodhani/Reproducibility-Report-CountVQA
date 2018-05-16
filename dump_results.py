import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import argparse
import base_model
from data_test import *

'''
Dumps the results into a JSON file for use by Official VQA evaluation script
'''

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_dev', action='store_true')
    parser.add_argument('--ckpt_file', type=str, required=True)

    args = parser.parse_args()

    try:
        assert ((args.val and not args.test and not args.test_dev) or (args.test and not args.val and not args.test_dev) or (args.test_dev and not args.test and not args.val))
    except:
        raise Exception('Only one of val, test, test-dev can be True')

    loader = get_loader(train=False, val=args.val, test=args.test, test_dev=args.test_dev)

    net = base_model.build_baseline0(loader.dataset.num_tokens, hid_dim=1024, v_dim=2048, n_objects=10, max_answers=3000, n_glimpse=2).cuda()
    net.load_state_dict(torch.load(args.ckpt_file)['weights'])

    net.eval()

    id_2_ques_id_map = loader.dataset.id_2_ques_id_map
    id_2_ans_map = loader.dataset.id_2_ans_map

    results = []

    print('Total n_batches = {}'.format(len(loader)))
    batch_id = 0

    for v, q, a, b, idx, q_len in loader:
        print(batch_id)
        var_params = {
            'volatile': True,
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)
        b = Variable(b.cuda(async=True), **var_params)
        q_len = Variable(q_len.cuda(async=True), **var_params)

        out = net(v, b, q, q_len)
        _, predicted_index = out.max(dim=1, keepdim=True)
        predicted_index = predicted_index.data.cpu()

        for i in range(idx.size(0)):
            results.append({'question_id': id_2_ques_id_map[idx[i]], 'answer': id_2_ans_map[predicted_index[i][0]]})

        batch_id += 1

    if args.val:
        result_filename = 'val_results.json'
    elif args.test:
        result_filename = 'test_results.json'
    elif args.test_dev:
        result_filename = 'test_dev_results.json'

    with open(result_filename,'w') as f1:
        json.dump(results, f1, indent=1)




