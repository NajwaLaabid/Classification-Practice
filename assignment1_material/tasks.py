import numpy

import resources
import pandas as pd
# import pdb ### useful to do debugging and step-by-step execution

TASK1 = False
TASK2 = True
TASK3 = False
        
if TASK1:
    dataset = numpy.array([   
        [1, 4, -1, 0, 0],
        [3, 6, -2, 0, 0],
        [7, 5, -6, 0, 0],
        [2, 5, 1, 0, 1],
        [0, 4, 6, 0, 1],
        [4, 6, 2, 0, 1],
        [6, 2, -1, 0, 1],
        [8, 3, -6, 0, 1],
        [7, 1, 1, 1, 2],
        [3, 2, 6, 1, 2],
        [5, 1, 2, 1, 2],
        [1, 3, 2, 1, 2]])
    
    # algo_params = {"split_measure": resources.gini, "max_depth": 5, "min_size": 0}
    algo_params = {"split_measure": resources.information_gain, "max_depth": 5, "min_size": 0}

    tree = resources.build_tree(dataset, **algo_params)
    print(resources.disp_tree(tree))
        
    test_set = numpy.array([
        [ 7, 4, -6, 0, 0],
        [ 3, 3, 1, 0, 0],
        [ 4, 5, -2, 0, 1],
        [ 6, 5, 6, 1, 1],
        [ 0, 3, -1, 0, 1],
        [ 9, 2, 1, 1, 1],
        [ 1, 1, 6, 0, 2]])
    
    # tree = {'index': 1, 'value': 4,
    #             'right': {'index': 2, 'value': 1, 'right': (1, 1.0), 'left': (0, 1.0)}, 
    #             'left': {'index': 2, 'value': 1, 'right': (2, 1.0), 'left': (1, 1.0)}}
    # ys, confs = zip(*[resources.tree_predict_row(test_set[i,:], tree) for i in range(test_set.shape[0])])
    # predicted, conf = numpy.array(ys), numpy.array(confs)
    # score = resources.accuracy_metric(test_set[:, -1], predicted)

if TASK2:
    data_params = {"filename": "iris-SV-sepal.csv", "last_column_str": True}
    ratio_train = .8
    nb_repeat = 1
    algo_params = {"split_measure": resources.information_gain, "max_depth": 5, "min_size": 3}

    dataset, head, classes = resources.load_csv(**data_params)
    print(classes)
    for fi in range(nb_repeat):
        ids = numpy.random.permutation(dataset.shape[0])
        split_pos = int(len(ids)*ratio_train)
        train_ids, test_ids = ids[:split_pos], ids[split_pos:]
        train_set = dataset[train_ids]
        test_set = dataset[test_ids]

        print('train ids: ', train_ids)
        print('test ids: ', test_ids)

        tree = resources.build_tree(train_set, **algo_params)
        ys, confs = zip(*[resources.tree_predict_row(test_set[i,:], tree) for i in range(test_set.shape[0])])
        
        print("Run #%d" % fi)
        bin_lbls = 1*(test_set[:,-1]==1)
        bin_pred = 1*(numpy.array(ys)==1)
        stat, cm = resources.get_CM_vals(bin_lbls, bin_pred)
        print("Stats", stat)
        print("Confusion matrix", cm)
    
if TASK3:
    for filename, c in [("iris-SV-sepal.csv", 0), ("iris-VV-length.csv", 2)]:
        ratio_train = .8
        data_params = {"filename": filename, "last_column_str": True}
        algo_params = {"c": c, "ktype": "linear", "kparams": {}}
        dataset, head, classes = resources.load_csv(**data_params)
        
        ids = numpy.random.permutation(dataset.shape[0])
        split_pos = int(len(ids)*ratio_train)
        train_ids, test_ids = ids[:split_pos], ids[split_pos:]
        train_set = dataset[train_ids]
        test_set = dataset[test_ids]

        model, svi = resources.prepare_svm_model(train_set[:,:-1], train_set[:,-1], **algo_params)        
        t = resources.svm_predict_vs(test_set[:,:-1], model)
        predicted = (numpy.sign(t)+1)/2.
        score = resources.accuracy_metric(test_set[:, -1], predicted)
        print('accuracy score: ', score)
        ### visu_plot_svm
        resources.visu_plot_svm(train_set, test_set, model)
        
