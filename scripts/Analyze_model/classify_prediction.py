import numpy as np
import sys, os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='classify_predictions',
                    description='Classifies data points based on two cutoffs')
    parser.add_argument('datafile1', type=str)
    parser.add_argument('datafile2', type=str)
    parser.add_argument('cutoff1', type=float)
    parser.add_argument('cutoff2', type=float)
    
    parser.add_argument('--column1', type = int, default = -1)
    parser.add_argument('--column2', type = int, default = -1)
    
    args = parser.parse_args()

    measured = np.genfromtxt(args.datafile1, dtype = str)
    predictions = np.genfromtxt(args.datafile2, dtype = str)

    mcolum = args.column1
    pcolum = args.column2

    cut_meas = args.cutoff1
    cut_pred = args.cutoff2

    outname = os.path.splitext(args.datafile2)[0]+ '_'+str(pcolum)+'_eval_on_'+os.path.splitext(os.path.split(args.datafile1)[1])[0]+'_'+str(mcolum) +'_cut'+ str(cut_meas) +'_and_'+ str(cut_pred) + '.txt'

    msort = np.argsort(measured[:,0])[np.isin(np.sort(measured[:,0]), predictions[:,0])]
    measured = measured[msort]

    psort = np.argsort(predictions[:,0])[np.isin(np.sort(predictions[:,0]), measured[:,0])]
    predictions = predictions[psort]

    preds = np.absolute(predictions[:,pcolum].astype(float)) > cut_pred
    meas= np.absolute(measured[:,mcolum].astype(float)) > cut_meas
    print(int(np.sum(preds)), 'predicted >', cut_pred)
    print(int(np.sum(meas)), 'measured >', cut_meas)
    sign = np.sign(measured[:,mcolum].astype(float)*predictions[:,pcolum].astype(float)) > 0
    correct = preds * meas * sign

    correctlist = measured[correct,0]
    print('Both correctly predicted', len(correctlist))
    np.savetxt(outname, correctlist, fmt = '%s')

    for i in np.where(correct)[0]:
        print(measured[i,0], measured[i,mcolum],predictions[i,pcolum])

