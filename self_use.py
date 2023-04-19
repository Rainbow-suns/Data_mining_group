print("Done!")
print("VGG training time: 0:12:20.778722.")


algorithm = 'as'
lr = 1e-4
bs = 8
print('./TrainResults/Result for ' + algorithm + "_lr_%.5f_bs_%d" % (lr, bs))

print('VGGnet training time: 0:41:42.048558. \nUsing cuda device \nAccuracy: 0.845 \nPrecison: 0.859 \nRecall: 0.836 '
      '\nF1Score: 0.847 \nConfusion matrix, without normalization \n[[662. 113.] \n    [135. 687.]] \nBest '
      'result:0.896 \nBest parameters:{\'lr\': 0.00009, \'bs\': 6}')