import csv
import numpy
import matplotlib.pyplot as plt
import copy 
import pandas as pd
import matplotlib.gridspec as gridspec

VERBOSE_TREE = False
CLASSES = {
    'blue': 0,
    'red': 1,
    'yellow': 2
}

# Load a CSV file
def load_csv(filename, last_column_str=False, normalize=False, as_int=False):
    dataset = list()
    head = None
    classes = {}
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for ri, row in enumerate(csv_reader):
            if not row:
                continue
            if ri == 0:
                head = row
            else:
                rr = [r.strip() for r in row]
                if last_column_str:
                    if rr[-1] not in classes:
                        classes[rr[-1]] = len(classes)
                    rr[-1] = classes[rr[-1]]
                dataset.append([float(r) for r in rr])
    dataset = numpy.array(dataset)
    if not last_column_str and len(numpy.unique(dataset[:,-1])) <= 10:
        classes = dict([("%s" % v, v) for v in numpy.unique(dataset[:,-1])])
    if normalize:
        dataset = normalize_dataset(dataset)
    if as_int:
        dataset = dataset.astype(int)
    return dataset, head, classes

# Find the min and max values for each column
def dataset_minmax(dataset):
    return numpy.vstack([numpy.min(dataset, axis=0), numpy.max(dataset, axis=0)])

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax=None):
    if minmax is None:
        minmax = dataset_minmax(dataset)
    return (dataset - numpy.tile(minmax[0, :], (dataset.shape[0], 1))) / numpy.tile(minmax[1, :]-minmax[0, :], (dataset.shape[0], 1))

# Sample k random points from the domain 
def sample_domain(k, minmax=None, dataset=None):
    if dataset is not None:
        minmax = dataset_minmax(dataset)
    if minmax is None:
        return numpy.random.random(k)
    d = numpy.random.random((k, minmax.shape[1]))
    return numpy.tile(minmax[0, :], (k, 1)) + d*numpy.tile(minmax[1, :]-minmax[0, :], (k, 1))

# Compute distances between two sets of instances
def euclidean_distance(A, B):
    return numpy.vstack([numpy.sqrt(numpy.sum((A - numpy.tile(B[i,:], (A.shape[0], 1)))**2, axis=1)) for i in range(B.shape[0])]).T

def L1_distance(A, B):
    return numpy.vstack([numpy.sum(numpy.abs(A - numpy.tile(B[i,:], (A.shape[0], 1))), axis=1) for i in range(B.shape[0])]).T

# Calculate contingency matrix
def contingency_matrix(actual, predicted, weights=None):
    if weights is None:
        weights = numpy.ones(actual.shape[0], dtype=int)
    ac_int = actual.astype(int)
    prd_int = predicted.astype(int)
    ## 3 d matrix to count in two ways: one with weight and one without weight
    counts = numpy.zeros((numpy.maximum(2,numpy.max(prd_int)+1), numpy.maximum(2,numpy.max(ac_int)+1), 2), dtype=type(weights[0]))
    for p,a,w in zip(prd_int, ac_int, weights):
        counts[p, a, 0] += 1
        counts[p, a, 1] += w
    return counts

# Calculate metrics from confusion matrix
def TPR_CM(confusion_matrix):
    if confusion_matrix[1,1] == 0: return 0.
    return (confusion_matrix[1,1])/float(confusion_matrix[1,1]+confusion_matrix[0,1])
def TNR_CM(confusion_matrix):
    if confusion_matrix[0,0] == 0: return 0.
    return (confusion_matrix[0,0])/float(confusion_matrix[0,0]+confusion_matrix[1,0])
def FPR_CM(confusion_matrix):
    if confusion_matrix[1,0] == 0: return 0.
    return (confusion_matrix[1,0])/float(confusion_matrix[0,0]+confusion_matrix[1,0])
def FNR_CM(confusion_matrix):
    if confusion_matrix[0,1] == 0: return 0.
    return (confusion_matrix[0,1])/float(confusion_matrix[0,1]+confusion_matrix[1,1])
def recall_CM(confusion_matrix):
    return TPR_CM(confusion_matrix)
def precision_CM(confusion_matrix):
    if confusion_matrix[1,1] == 0: return 0.
    return (confusion_matrix[1,1])/float(confusion_matrix[1,0]+confusion_matrix[1,1])
def accuracy_CM(confusion_matrix):
    if confusion_matrix[0,0] == 0 and confusion_matrix[1,1] == 0: return 0.
    return (confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix[0,0]+confusion_matrix[1,1]+
    confusion_matrix[0,1]+confusion_matrix[1,0])
metrics_cm = {"TPR": TPR_CM, "TNR": TNR_CM, "FPR": FPR_CM, "FNR": FNR_CM,
              "recall": recall_CM, "precision": precision_CM, "accuracy": accuracy_CM}
    
def get_CM_vals(actual, predicted, weights=None, vks=None):
    if vks is None:
        vks = metrics_cm.keys()
    cm = contingency_matrix(actual, predicted, weights)
    if weights is None:
        cm = cm[:, :, 0]
    else:
        cm = cm[:, :, 1]
    vals = {}
    for vk in vks:
        if vk in metrics_cm:
            vals[vk] = metrics_cm[vk](cm)
    return vals, cm

# Calculate the error rate for a split dataset
def error_rate(groups, lbls, weights=None):
    er = 0
    nb_instances = 0.
    for g in groups:
        if len(g) > 0:
            if weights is not None:
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                cs = numpy.bincount(idx, minlength=len(weights))*weights.astype(float)
                nb_winstances = numpy.sum(cs)
                er += nb_winstances-numpy.max(cs)
                nb_instances += nb_winstances
            else:
                nb_instances += len(g)
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                cis = numpy.bincount(idx)
                er += len(g)-numpy.max(cis)
    return er/nb_instances

# Calculate the entropy for a split dataset
def entropy(groups, lbls, weights=None):
    entropy = 0.
    nb_instances = 0.
    for g in groups: # two groups, yes and no
        if len(g) > 0:
            if weights is not None:
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                cs = numpy.bincount(idx, minlength=len(weights))*weights.astype(float)
                nb_winstances = numpy.sum(cs)
                pis = cs/nb_winstances
                nb_instances += nb_winstances
                entropy -= nb_winstances*numpy.sum(pis*numpy.log2(pis))
            else:
                nb_instances += len(g)
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                occ = numpy.bincount(idx)
                pis = (occ/len(g))
                entropy += len(g)*numpy.sum(-pis*numpy.log2(pis))

    return entropy/nb_instances

# Calculate the the information gain for a split dataset
def information_gain(groups, lbls, weights=None):
    ce = entropy(groups, lbls, weights)
    pe = entropy([numpy.hstack(groups)], lbls, weights)
    return pe-ce

# Calculate the Gini index for a split dataset
def gini(groups, lbls, weights=None):
    gini = 0.
    nb_instances = 0.
    for g in groups:
        if len(g) > 0:
            if weights is not None:
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                cs = numpy.bincount(idx, minlength=len(weights))*weights.astype(float)
                nb_winstances = numpy.sum(cs)
                pis = cs/nb_winstances
                nb_instances += nb_winstances
                gini += nb_winstances*(1 - numpy.sum(pis**2))
            else:
                nb_instances += len(g)
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                pis = numpy.bincount(idx)/float(idx.shape[0])
                gini += idx.shape[0]*(1 - numpy.sum(pis**2))
    return gini/nb_instances

# Calculate accuracy percentage
def accuracy_metric(actual, predicted, weights=None):
    if weights is not None:
        ac_int = actual.astype(int)
        return numpy.sum(weights[ac_int[actual == predicted]])/float(numpy.sum(weights[ac_int]))
    return numpy.sum(actual == predicted)/float(actual.shape[0])

########################################################
#### DECISION TREE CLASSIFIER
########################################################

# Split a dataset based on an attribute and an attribute value
def test_split(data, dindices, vindex, value):
    mask = data[dindices, vindex] < value
    return dindices[mask], dindices[numpy.logical_not(mask)]

# Select the best split point for a dataset
def get_split(data, dindices, split_measure, weights=None, min_size=-1):
    candidates = []
    lbls = data[:, -1].astype(int)
    ll = max(2, len(numpy.unique(lbls)))
    sc = split_measure([dindices], lbls, weights=weights)
    best = {'index': None, 'value': None, 'groups': [dindices, dindices[[]]], "score": sc}
    logs = []
    for vindex in range(data.shape[1]-1): ## iterate over attributes
        splitvs = numpy.unique(data[dindices, vindex]) ## get all unique values for a given attribute
        for si in range(splitvs.shape[0]-1):
            #splitv = (splitvs[si]+splitvs[si+1])/2. ## one possible discretization method 
            splitv = splitvs[si+1]
            groups = test_split(data, dindices, vindex, splitv)
            # if any([len(g) < min_size for g in groups]): ## stop at a given leaf size
            #     continue
            score = split_measure(groups, lbls, weights=weights)
            candidates.append({'index': vindex, 'value': splitv, 'groups': groups, "score": score})
            if VERBOSE_TREE:
                # g0 = "&\\textcolor{TolBlue}{%d} & \\textcolor{TolRed}{%d} & \\textcolor{TolYellow}{%d}&" % tuple(numpy.bincount(lbls[groups[0]], minlength=3))
                # g1 = "&\\textcolor{TolBlue}{%d} & \\textcolor{TolRed}{%d} & \\textcolor{TolYellow}{%d}&" % tuple(numpy.bincount(lbls[groups[1]], minlength=3))
                # lstr = "$v_%d \\geq %s$ & %s & %s & $%.4f$ \\\\" % (vindex+1, splitv, g0, g1, -score)
                gs = " ".join(["/".join(["%d" % d for d in numpy.bincount(lbls[g], minlength=ll)]) for g in groups])
                lstr = "v_%d \\geq %s\t%s\t%.4f" % (vindex+1, splitv, gs, -score)
                logs.append((score, lstr))
            if best is None or score > best["score"]:
                best = {'index': vindex, 'value': splitv, 'groups': groups, "score": score}

    if VERBOSE_TREE:
        if best is not None and best["index"] is not None:
            # for (sc, l) in sorted(logs):
            for (sc, l) in logs:
                print(l)
            g_counts =  (numpy.bincount(lbls[best["groups"][0]], minlength=ll), numpy.bincount(lbls[best["groups"][1]], minlength=ll))
            # g_points =  (", ".join(["%d" % ki for ki in best["groups"][0]+1]), ", ".join(["%d" % ki for ki in best["groups"][1]+1]))
            g_points =  (" ", " ")
            print("<<< MADE SPLIT $v_%d >= %s$ no/L:%d=%s {%s} yes/R:%d=%s {%s} %s" % (best["index"]+1, best["value"], len(best["groups"][0]), g_counts[0], g_points[0], len(best["groups"][1]), g_counts[1], g_points[1], best["score"]))
        else:
            print("<<< DID NOT SPLIT")
    return best, candidates

# Create a terminal node value
def to_terminal(data, dindices, weights=None):
    tst = data[dindices, -1].astype(int)
    nbdv = numpy.unique(data[:, -1]).shape[0]
    if weights is None:
        ps = numpy.bincount(tst, minlength=nbdv)/float(tst.shape[0])
    else:
        ps = numpy.bincount(tst, minlength=nbdv)*weights.astype(float)
        ps /= numpy.sum(ps)
    top = numpy.argmax(ps)
    return (top, ps[top])

# Create child splits for a node or make terminal
def split(data, node, split_measure, max_depth, min_size, depth, root=None, weights=None, steps=[], cur_candidates=[], cnt=0):
    # log every run of the construction of the tree 
    steps.append(
        {
            'cnt': cnt,
            'split_candidates': cur_candidates,
            'best_split': node
        }
    )
    
    if node is None:
        return None
    left, right = node['groups']
    both_terminals = True
    # del(node['groups'])
    # check for a no split 
    if len(left) == 0 or len(right) == 0: ## => check for leaf
        node['left'] = node['right'] = to_terminal(data, numpy.hstack([left, right]), weights=weights)
        return node['right']
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(data, left, weights=weights), to_terminal(data, right, weights=weights)
        if node['left'][0] == node['right'][0]:
            return to_terminal(data, numpy.hstack([left, right]), weights=weights)
        return None
    # process left child
    if len(left) <= min_size or len(set(data[left,-1])) == 1: ## if less than leaf_size or pure leaf (single class)
        node['left'] = to_terminal(data, left, weights=weights)
    else:
        node['left'], left_candidates = get_split(data, left, split_measure, weights=weights, min_size=min_size)
        if node['left'] is None:
            node['left'] = to_terminal(data, left, weights=weights)
        else:
            ret = split(data, node['left'], split_measure, max_depth, min_size, depth+1, root=root, weights=weights, steps=steps, cur_candidates=left_candidates, cnt=cnt+1)
            if ret is not None:
                node['left'] = ret
            else:
                both_terminals = False
    # process right child
    if len(right) <= min_size or len(set(data[right,-1])) == 1:
        node['right'] = to_terminal(data, right, weights=weights)
    else:
        node['right'], right_candidates = get_split(data, right, split_measure, weights=weights, min_size=min_size)
        if node['right'] is None:
            node['right'] = to_terminal(data, right, weights=weights)
        else:
            ret = split(data, node['right'], split_measure, max_depth, min_size, depth+1, root=root, weights=weights, steps=steps, cur_candidates=right_candidates, cnt=cnt+1)
            if ret is not None:
                node['right'] = ret
            else:
                both_terminals = False
            
    if both_terminals and node['right'][0] == node['left'][0]:
        return to_terminal(data, numpy.hstack([left, right]), weights=weights)
    return None
        
def disp_tree(node, depth=0, side=" "):
    map_sides = {"l": "no ", "r": "yes"}
    sss = "%s%s |_ [v%d >= %s] score=%.3f\n" % (depth*"\t", map_sides.get(side,""), node['index']+1, node['value'], node['score'])
    for sd, gi in [("right", 1), ("left", 0)]:
        if sd in node:
            if isinstance(node[sd], dict):
                sss += disp_tree(node[sd], depth+1, sd[0])
            else:
                sss += "%s%s |_  y=%s #%d\n" % ((depth+1)*"\t", map_sides.get(sd[0],""), node[sd], len(node["groups"][gi]))
    return sss

## creates a pandas data frame from the list of candidates
def get_labels_group(lbls, group):
    str_group = ''
    for c in CLASSES:
        lbls_c = [(point+1) for idx, point in enumerate(group) if lbls[point] == CLASSES[c]]
        str_group += c + ': ' + str(lbls_c) + '\n'

    return str_group

def get_candidates_df(data, candidates):
    lbls = data[:,-1].astype(int)
    candidate_rows = []
    for c in candidates:
        split_test = 'v' + str(c['index']+1) + '>=' + str(c['value'])
        no_subset  = get_labels_group(lbls, c.get('groups')[0])
        yes_subset = get_labels_group(lbls, c.get('groups')[1])
        ig = round(c['score'], 4)
        candidate_rows.append({
            'split test': split_test,
            'no subset': no_subset,
            'yes subset': yes_subset,
            'IG': ig
        })
        
    df = pd.DataFrame(candidate_rows)

    return df

# display the steps of construction of the decision tree
def disp_steps(data, steps=[]):
    n_steps = len(steps)
    # print steps
    for i in range(n_steps):
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')  
        df = get_candidates_df(data,steps[i]['split_candidates'])
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        table.scale(2,5)
        #plt.savefig("tablepdf%d.pdf" % i, bbox_inches='tight')
    
# Build a decision tree
def build_tree(data, split_measure, max_depth, min_size, weights=None):
    root, cur_candidates = get_split(data, numpy.arange(data.shape[0]), split_measure, weights=weights)
    steps = []
    split(data, root, split_measure, max_depth, min_size, 1, root=root, weights=weights,  steps=steps, cur_candidates=cur_candidates, cnt=1)
    disp_steps(data,steps)
    return root

# Make a prediction with a decision tree
def tree_predict_row(row, node):
    if row[node['index']] < node['value']:
        if "left" not in node:
            return (.5, .5)
        if isinstance(node['left'], dict):
            return tree_predict_row(row, node['left'])
        else:
            return node['left']
    else:
        if "right" not in node:
            return (.5, .5)
        if isinstance(node['right'], dict):
            return tree_predict_row(row, node['right'])
        else:
            return node['right']        

########################################################
#### SUPPORT VECTOR MACHINES (SVM)
########################################################
import cvxopt.solvers
MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class Kernel(object):
    
    kfunctions = {}
    kfeatures = {}
    
    def __init__(self, ktype='linear', kparams={}):
        self.ktype = 'linear'
        if ktype in self.kfunctions:
            self.ktype = ktype
        else:
            raise Warning("Kernel %s not implemented!" % self.ktype)
        self.kparams = kparams

    def distance_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        return self.kfunctions[self.ktype](X, Y, **self.kparams)

def linear(X, Y):
    return numpy.dot(X, Y.T)
Kernel.kfunctions['linear'] = linear

def polynomial(X, Y, degrees=None, offs=None):
    if degrees is None:
        return linear(X, Y)
    if offs is None:
        return numpy.sum(numpy.dstack([numpy.dot(X, Y.T)**d for d in degrees]), axis=2)
    return numpy.sum(numpy.dstack([(numpy.dot(X, Y.T)+offs[i])**d for i,d in enumerate(degrees)]), axis=2)
Kernel.kfunctions['polynomial'] = polynomial

def RBF(X, Y, sigma):
    return numpy.vstack([numpy.exp(-numpy.sum((X-numpy.outer(numpy.ones(X.shape[0]), Y[yi,:]))** 2, axis=1) / (2. * sigma ** 2)).T for yi in range(Y.shape[0])]).T
Kernel.kfunctions['RBF'] = RBF

def compute_multipliers(X, y, c, kernel):
    n_samples, n_features = X.shape
    
    K = kernel.distance_matrix(X)
    P = cvxopt.matrix(numpy.outer(y, y) * K)
    q = cvxopt.matrix(-1 * numpy.ones(n_samples))
    ### hard margin: c == 0
    if c == 0:
        G = cvxopt.matrix(numpy.eye(n_samples)*-1)
        h = cvxopt.matrix(numpy.zeros(n_samples))
    else:
        G = cvxopt.matrix(numpy.vstack((numpy.eye(n_samples)*-1, numpy.eye(n_samples))))       
        h = cvxopt.matrix(numpy.hstack((numpy.zeros(n_samples), numpy.ones(n_samples) * c)))
    A = cvxopt.matrix(numpy.array([y]), (1, n_samples))
    b = cvxopt.matrix(0.0)
    
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    return numpy.ravel(solution['x'])

def svm_predict_vs(data, model):
    xx = model["kernel"].distance_matrix(model["support_vectors"], data)
    yy = model["lmbds"] * model["support_vector_labels"]
    return model["bias"] + numpy.dot(xx.T, yy)

def prepare_svm_model(X, y, c, ktype="linear", kparams={}):
    ### WARNING: work with labels {-1, 1} !
    y = 2.*y-1
    kernel = Kernel(ktype, kparams)
        
    lagrange_multipliers = compute_multipliers(X, y, c, kernel)
    support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
    
    model = {"kernel": kernel, "bias": 0.0,
             "lmbds": lagrange_multipliers[support_vector_indices],
             "support_vectors": X[support_vector_indices],
             "support_vector_labels": y[support_vector_indices]}
    pvs = svm_predict_vs(model["support_vectors"], model)
    #model["bias"] = -pvs[0]
    ### ... bias = -(max prediction for positive support vector + min prediction for positive support vector)/2
    model["bias"] = -(numpy.max(pvs)+numpy.min(pvs))/2.
    return model, support_vector_indices


def visu_plot_svm(train_set, test_set, model, svi=None):
    minmax = dataset_minmax(numpy.vstack([train_set, test_set]))
    i, j = (0,1)
    gs = []
    lims = []
    for gi in range(train_set.shape[1]):
        step_size = float(minmax[1, gi]-minmax[0, gi])/100
        gs.append(numpy.arange(minmax[0, gi]-step_size, minmax[1, gi]+1.5*step_size, step_size))
        lims.append([minmax[0, gi]-2*step_size, minmax[1, gi]+2*step_size])
    axe = plt.subplot()
    
    bckgc = (0, 0, 0, 0)
    color = "#888888"
    color_lgt = "#DDDDDD"
    cmap="coolwarm"
    ws = numpy.dot(model["lmbds"]*model["support_vector_labels"], model["support_vectors"])
    coeffs = numpy.hstack([ws, [model["bias"]]])
    sv_points = []

    ## print coeffs => equation of hyperplane
    eq = [round(c, 2) for c in coeffs]
    print('eq: ', eq)

    vmin, vmax = (minmax[0,-1], minmax[1,-1])
    axe.scatter(train_set[:, j], train_set[:,i], c=train_set[:,-1], vmin=vmin, vmax=vmax, cmap=cmap, s=50, marker=".", edgecolors='face', linewidths=2, gid="data_points_lbl")
    axe.scatter(test_set[:, j], test_set[:,i], c=test_set[:,-1], vmin=vmin, vmax=vmax, cmap=cmap, s=55, marker="*", edgecolors='face', linewidths=2, zorder=2, gid="data_points_ulbl")

    
    if svi is not None:
        sv_points = numpy.where(svi)[0]

    xs = numpy.array([gs[j][0],gs[j][-1]])
    tmp = -(coeffs[i]*numpy.array([gs[i][0],gs[i][-1]])+coeffs[-1])/coeffs[j]
    xtr_x = numpy.array([numpy.maximum(tmp[0], xs[0]), numpy.minimum(tmp[-1], xs[-1])])

    x_pos = xtr_x[0]+.1*(xtr_x[1]-xtr_x[0])
    x_str = xtr_x[0]+.66*(xtr_x[1]-xtr_x[0])
    
    ys = -(coeffs[j]*xs+coeffs[-1])/coeffs[i]
    axe.plot(xs, ys, "-", color=color, linewidth=0.5, zorder=5, gid="details_svm_boundary")
        
    closest = (None, 0.)
    p0 = numpy.array([0, -coeffs[-1]/coeffs[i]])
    ff = numpy.array([1, -coeffs[j]/coeffs[i]])
    V = numpy.outer(ff, ff)/numpy.dot(ff, ff)
    offs = numpy.dot((numpy.eye(V.shape[0]) - V), p0)

    mrgs = [1.]
    for tii, ti in enumerate(sv_points):
        proj = numpy.dot(V, train_set[ti,[j, i]]) + offs
        axe.plot([train_set[ti,j], proj[0]], [train_set[ti,i], proj[1]], color=color_lgt, linewidth=0.25, zorder=0, gid=("details_svm_sv%d" % tii))
            
    #### plot margin
    for mrg in mrgs:
         yos = -(coeffs[j]*xs+coeffs[-1]-mrg)/coeffs[i]
         axe.plot(xs, yos, "-", color=color_lgt, linewidth=0.5, zorder=0, gid="details_svm_marginA")
         yos = -(coeffs[j]*xs+coeffs[-1]+mrg)/coeffs[i]
         axe.plot(xs, yos, "-", color=color_lgt, linewidth=0.5, zorder=0, gid="details_svm_marginB")

         mrgpx = x_pos # xs[0]+pos*(xs[-1]-xs[0])
         mrgpy = -(coeffs[j]*mrgpx+coeffs[-1]-mrg)/coeffs[i]
         mrgp = numpy.array([mrgpx, mrgpy])
         proj = numpy.dot(V, mrgp) + offs
         mrgv = numpy.sqrt(numpy.sum((mrgp-proj)**2))
         print("m=%.3f" % mrgv)
         axe.arrow(proj[0], proj[1], (proj[0]-mrgpx), (proj[1]-mrgpy), length_includes_head=True, color=color, linewidth=0.25, gid="details_svm_mwidthA")
         axe.arrow(proj[0], proj[1], -(proj[0]-mrgpx), -(proj[1]-mrgpy), length_includes_head=True, color=color, linewidth=0.25, gid="details_svm_mwidthB")

         
         axe.annotate("m=%.3f" % mrgv, (proj[0], proj[1]), (0, 15), textcoords='offset points', color=color, backgroundcolor=bckgc, zorder=10, gid="details_svm_mwidth-ann")
    axe.set_xlim(lims[j][0], lims[j][-1])
    axe.set_ylim(lims[i][0], lims[i][-1])
    plt.show()
