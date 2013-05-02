# ----------------------------------------------------
# Command line interface for the Neural Representation Benchmark
# ----------------------------------------------------
# for help:
# python commandline_interface.py --help


# Kernel analysis code derived from:
# #### some additions and optimizations
# ####  by Dan Yamins, Charles Cadieu, Nicolas Pinto
# ----------------------------------------------------
# Kernel analysis of a representation
# ----------------------------------------------------
#
# Copyright: Gregoire Montavon
#
# This code is released under the MIT licence:
# http://www.opensource.org/licenses/mit-license.html
#
# ----------------------------------------------------
#
# This code is similar to the analysis described in:
#
#   G. Montavon, M. Braun, K.-R. Mueller
#   Kernel Analysis of Deep Networks
#   2011, Journal of Machine Learning Research (JMLR)
#
#   M. Braun, J. Buhmann, K.-R. Mueller
#   On Relevant Dimensions in Kernel Feature Spaces
#   2008, Journal of Machine Learning Research (JMLR)
#
# ----------------------------------------------------

from collections import defaultdict

import numpy as np
import scipy as sp
from scipy import linalg


class KPCAValueError(Exception):
    pass


def kanalysis_predict(X, T, Xpred, Tpred, Q=[0.1, 0.5, 0.9], nullspace_correction=True):
    curves, curves_pred, var_max_idxs = kanalysis_predict_core(X, T, Xpred, Tpred, Q,
                                                               nullspace_correction=nullspace_correction)
    return curves, curves_pred, var_max_idxs


def kanalysis_predict_core(X, T, Xpred, Tpred,
                           Q=[0.1, 0.5, 0.9],
                           nullspace_correction=True):

    # Ensure T is for multiclass:
    if T.ndim == 1:
        T = T[:, np.newaxis]

    # Check dims and shapes:
    assert X.ndim == 2, X.ndim
    assert T.ndim == 2, T.ndim
    assert X.shape[0] == T.shape[0], (X.shape[0], T.shape[0])

    # Ensure Tpred is for multiclass:
    if Tpred.ndim == 1:
        Tpred = Tpred[:, np.newaxis]

    # Check dims and shapes:
    assert Xpred.ndim == 2, Xpred.ndim
    assert Tpred.ndim == 2, Tpred.ndim
    assert Xpred.shape[0] == Tpred.shape[0], (Xpred.shape[0], Tpred.shape[0])

    # Ensure that we operate on float64
    X = X.astype(np.float64)  # needs to be double precision
    Xpred = Xpred.astype(np.float64)

    # Computing pairwise distances
    X2sum1 = (X ** 2).sum(1)
    D2 = X2sum1[:, np.newaxis] + X2sum1[np.newaxis, :] - 2 * np.dot(X, X.T)
    # compute rank ordered distances to choose sigma:
    DF = D2.flatten()
    DF.sort()

    # Computing pairwise distances for Xpred
    Xpred2 = X2sum1[:, np.newaxis] + (Xpred ** 2).sum(1)[np.newaxis, :] - 2 * np.dot(X, Xpred.T)

    # Centering labels. Normalizes squared loss to be between 0 and 1 on training set. (can be greater on testing)
    Tmean = T.mean(axis=0)
    T = T - Tmean
    Tstd = T.std()
    T = T / Tstd

    # init stored outputs:
    curves = []
    Ppreds = []
    var_max_idxs = []
    # loop over Q:
    for q in Q:

        # choose a value of sig
        sig = DF[int(q * len(DF))]
        # Computing the kernel
        print ('Computing the kernel')
        K = np.exp(-D2 / sig)

        K0mean = K.mean(0)[np.newaxis, :]
        K1mean = K.mean(1)[:, np.newaxis]
        Kmean = K.mean()

        # normalize the kernel:
        K = K - K0mean - K1mean + Kmean

        # normalize the embedding kernel:
        PointsK = np.exp(-Xpred2 / sig) - K0mean.T - K1mean + Kmean

        # Computing and sorting the kernel principal components
        print ('Computing the PCA')
        try:
            W, V = sp.linalg.eigh(K)
        except ValueError:
            raise KPCAValueError()
        V = V[:, np.argsort(-np.abs(W))]
        W = W[np.argsort(-np.abs(W))]

        # keep the full eigen space but not more
        # (this is important when K is *not* full rank)
        var_explained = (W ** 2.).cumsum() / (W ** 2.).sum()
        var_max_idx = var_explained.argmax()
        print 'For %2.2f variance explained need %d / %d eigenvectors' % \
              (100. * var_explained[var_max_idx], var_max_idx + 1, len(W))

        var_max_idxs.append(var_max_idx + 1)

        # if we correct the eigenspace, set null space to 0.
        if nullspace_correction:
            V[:, var_max_idx + 1:] = 0.

        # Projection of labels on the leading kernel principal components
        print ('Projecting')
        # compute the soluton: Theta
        Theta = np.dot(V.T, T)[:, np.newaxis]
        P = V.T[:, :, np.newaxis] * Theta  # predictions per eigen-component
        P = P.cumsum(0)  # cumsum to evaluate T estimate up to d eigen-components
        P = np.row_stack([np.zeros((1, ) + P.shape[1:]), P])  # P is zero for no eigenvalues
        Z = ((P - T) ** 2).mean(1)  # Squared loss function, averaged over points (now D by # classes)

        # compute the embedding kernel's projection into the eigen space:
        # we normalize the eigenspectrum by lambda**-1.
        PointsKprime = np.dot(np.dot(V, np.diag(1. / (1e-7 + W))).T, PointsK)
        Ppred = PointsKprime[:, :, np.newaxis] * Theta  # predictions per eigen-component
        Ppred = Ppred.cumsum(0)  # cumsum to evaluate Tpred estimate up to d eigen-components
        Ppred = np.row_stack([np.zeros((1, ) + Ppred.shape[1:]), Ppred])  # Ppred is zero for no eigenvalues

        # store the values:
        curves += [Z]
        Ppreds += [Ppred]

    print 'Finishing...'
    # Training curves:
    curves = np.array(curves)  # convert to array
    curves = np.mean(curves, axis=2)  # mean over classes
    min_inds = np.argmin(curves, axis=0)  # find minimum sigmas
    curves = np.array([curves[min_inds[ind], ind] for ind in range(len(min_inds))])  # select min sigmas

    # Compute prediction curves:
    Ppreds = np.array(Ppreds)  # convert to array
    Ppreds = np.array([Ppreds[min_inds[ind], ind] for ind in range(len(min_inds))])  # min sigmas (based on curves)
    # normalize Tpred:
    Tpred = Tpred - Tmean
    Tpred /= Tstd
    curves_pred = ((Ppreds - Tpred[np.newaxis, :, :]) ** 2).mean(1).mean(1)  # average over points, classes

    return curves, curves_pred, var_max_idxs


variations = ('Variation00', 'Variation03', 'Variation06')

meta_info = {'Variation00': {'metafilename': 'Variation00_20110203_meta.txt',
                             'imagedir': 'Variation00_20110203'},
             'Variation03': {'metafilename': 'Variation03_20110128_meta.txt',
                             'imagedir': 'Variation03_20110128'},
             'Variation06': {'metafilename': 'Variation06_20110131_meta.txt',
                             'imagedir': 'Variation06_20110131'}}

category_map = {'Animals': 0,
                'Cars': 1,
                'Chairs': 2,
                'Faces': 3,
                'Fruits': 4,
                'Planes': 5,
                'Tables': 6}


def parse_meta_data(base_path):
    from collections import defaultdict
    import os

    assert os.path.exists(base_path)

    meta = defaultdict(dict)
    for variation in variations:
        metafilename = os.path.join(base_path, meta_info[variation]['metafilename'])
        assert os.path.exists(metafilename)
        image_ids = []
        categories = []
        splits = []
        with open(metafilename, 'r') as fh:
            for line in fh.readlines():
                line = line.strip('\n')
                split_line = line.split(' ')

                image_ids.append(split_line[0])
                categories.append(category_map[split_line[1]])
                splits.append([True if item is '1' else False for item in split_line[2:]])
        meta[variation]['image_ids'] = image_ids
        meta[variation]['categories'] = np.array(categories)
        meta[variation]['splits'] = np.vstack(splits).T
        print meta[variation]['splits'].shape, meta[variation]['splits'].dtype

    return meta


def parse_feature_data(base_path, extension='.txt'):
    import os

    assert os.path.exists(base_path)

    feature_dict = defaultdict(dict)
    for variation in variations:
        feature_filename = os.path.join(base_path, meta_info[variation]['imagedir'] + extension)

        image_ids = []
        features = []
        with open(feature_filename, 'r') as fh:
            for line in fh.readlines():
                line = line.strip('\n')
                split_line = line.split(' ')

                image_id = split_line[0]
                image_ids.append(image_id)

                feature = [float(s) for s in split_line[1:]]
                features.append(feature)

        features = np.array(features)

        print features.shape, features.dtype

        feature_dict[variation]['image_ids'] = image_ids
        feature_dict[variation]['features'] = features

    return feature_dict


def run_standard_protocol(meta, feature_dict):

    ka_results = {}
    for variation in variations:
        # reorder features:
        image_ids = meta[variation]['image_ids']
        features = []
        for image_id in image_ids:
            assert feature_dict[variation]['image_ids'].count(image_id) == 1
            idx = feature_dict[variation]['image_ids'].index(image_id)
            features.append(feature_dict[variation]['features'][idx,:])
        features = np.vstack(features)

        variation_results = []
        for split_ind, split in enumerate(meta[variation]['splits']):
            print split_ind
            X = features[split, :]
            Xpred = features[-split, :]
            category_split = meta[variation]['categories'][split]
            Y = -1. * np.ones((category_split.shape[0], len(category_map)), dtype=np.float64)
            for cat in range(len(category_map)):
                Y[category_split == cat, cat] = 1.
            category_split_pred = meta[variation]['categories'][-split]
            Ypred = -1. * np.ones((category_split_pred.shape[0], len(category_map)), dtype=np.float64)
            for cat in range(len(category_map)):
                Ypred[category_split_pred == cat, cat] = 1.

            ka, ka_pred, vmi = kanalysis_predict(X=X, T=Y, Xpred=Xpred, Tpred=Ypred)

            variation_results.append({'ka_curve': ka,
                                      'ka_mean': ka.mean(),
                                      'ka_pred_curve': ka_pred,
                                      'var_max_idxs': vmi})

        ka_results[variation] = variation_results

    for variation in variations:
        print '=' * 30
        print variation

        var_results = [1. - item['ka_mean'] for item in ka_results[variation]]
        print 'Kernel Analysis, mean KA-AUC:', np.mean(var_results), '(%2.2e)' % np.std(var_results)

        var_genearalization_results = np.array([1. - item['ka_pred_curve'] for item in ka_results[variation]])
        print var_genearalization_results.shape
        var_genearalization_results_mean = np.mean(var_genearalization_results, axis=0)
        argmax = np.argmax(var_genearalization_results_mean)
        ka_results[variation]['best_gen_acc'] = var_genearalization_results_mean[argmax]
        print 'Generalization Accuracy:', var_genearalization_results_mean[argmax], \
            '(%2.2e)' % np.std(var_genearalization_results[:, argmax])

    return ka_results


if __name__ == '__main__':

    # Set default paths here:
    META_DATA_PATH = None
    FEATURE_DATA_PATH = None

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--image_dir', type=str, dest='image_dir',
                        help='Path to NeuralRepresentationImages_7x7 directory, for example:\n'
                             '    /Users/cadieu/Downloads/NeuralRepresentationImages_7x7_20130408',)
    parser.add_argument('--feature_dir', type=str, dest='feature_dir',
                        help='Path to feature directory containing files for each variation level.\n'
                             'These files should be named: Variation00_20110203.txt,\n'
                             '                             Variation03_20110128.txt,\n'
                             '                             Variation06_20110131.txt\n'
                             'Each line in the file is the feature for an image, beginning with the image id:\n'
                             '    Variation00_20110203/b0c7d9523215b272249d84287e1d28d851275f4f.png -0.45 -0.74 -0.37\n'
                             '    Variation00_20110203/d0f6824ddcaf80456ceb0e86def346fa2ac97112.png -0.28 -0.31 -0.27',)

    args = parser.parse_args()
    image_dir = args.image_dir
    if image_dir is None:
        image_dir = META_DATA_PATH
    feature_dir = args.feature_dir
    if feature_dir is None:
        feature_dir = FEATURE_DATA_PATH

    meta = parse_meta_data(image_dir)

    feature_dict = parse_feature_data(feature_dir)

    run_standard_protocol(meta, feature_dict)

    # TODO: other args to add:
    # feature_data_extension
    # feature size (optional)
    # directory to place output (optional)

    # TODO:
    # save ka curve to txt file
    # print ka value
    # include V4 and IT curves in release, maybe models as well
