import numpy as np

def calculate_mean_variance(data):
    '''
    To be obtained from the training data. The expected shape is (number_frames x number_feats)
    '''
    return data.mean(axis=0), data.std(axis=0)
    

def apply_mean_variance_normalisation(data, datamean, datastd):
    '''
    To apply mean variance normalisation on data.
    '''
    # we need to add a dimension to the mean and std vectors to do this
    
    return (data - datamean) / datastd
    
    
    
def create_windowed_feats (data, leftcontext, rightcontext):
    contextWindowLength = leftcontext + rightcontext + 1
    data = data.T
    numrows, numcols = data.shape
    
    firstcol = data[:,0]
    firstcols = np.tile(firstcol[:,np.newaxis], (1, leftcontext))
    lastcol = data[:,-1]
    lastcols = np.tile(lastcol[:,np.newaxis], (1, rightcontext))
    biggerdatamatrix = np.concatenate((firstcols, data, lastcols), axis=1)
    
    for k in range(contextWindowLength):
        if k == 0 :
            windowed_feats = biggerdatamatrix[:,k:numcols+k]
        else :
            chopped = biggerdatamatrix[:,k:numcols+k]
            chopped.shape
            windowed_feats = np.concatenate ((windowed_feats,chopped), axis=0)
    
    return  windowed_feats.T


def convert_to_b1(features):
    # expects hml features with delta
    featslen = features.shape[1]/6
    print ("Number of features per ANF type is " + str(featslen))
    
    feats1 = ((13 * features[:, :featslen]) + (3 * features[:, featslen:2*featslen] ) + (3 * features[:, 2*featslen: 3*featslen]))/19       
    feats2 = ((13 * features[:, 3*featslen : 4*featslen]) + (3 * features[:, 4*featslen:5*featslen] ) + (3 * features[:, 5*featslen: 6*featslen]))/19
    print ("Converted to B1 features of size " + str(feats1.shape[0]) + " x " + str(feats1.shape[1] + feats2.shape[1])  + ".")
    return np.concatenate((feats1, feats2), axis=1)

def select_feats_from_hml(features, tagselected):
    # expects hml features with delta
    featslen = features.shape[1]/6
    if (tagselected == "h"):
	return np.concatenate((features[:,:featslen],features[:, 3*featslen : 4*featslen]), axis=1)
    if (tagselected == "m"):
	return np.concatenate((features[:, featslen: 2*featslen],features[:, 4*featslen : 5*featslen]), axis=1)
    if (tagselected == "l"):
	return np.concatenate((features[:, 2*featslen: 3*featslen],features[:, 5*featslen : 6*featslen]), axis=1)


def make_tensor_lstm_withoutDelta(feats, labels,indices, maxlength) :
    # for creating 3D tensor from DNN input shaped features
    # requires features and labels in matrix format and indices that indicates sentence boundaries
    # maxlength denotes the maximum length of input data
     
    infeatdim = feats.shape[0]/2
    feat_tensor = np.zeros((indices.shape[1], maxlength, infeatdim))
    labels_tensor = np.zeros((indices.shape[1], maxlength, labels.shape[0]) )

    for k in range(indices.shape[1]) :
        secfeats = np.array(feats[:infeatdim, indices[0,k] - 1 : indices[1,k]])
        seclabels = np.array(labels[:, indices[0,k] - 1 : indices[1,k]])
        feat_tensor[k,:,:] = np.concatenate((secfeats.T, np.zeros((maxlength - secfeats.shape[1], infeatdim))), axis=0)
        labels_tensor[k,:,:] = np.concatenate((seclabels.T, np.zeros((maxlength - seclabels.shape[1], seclabels.shape[0] ))), axis=0)

    return feat_tensor, labels_tensor


def make_tensor_lstm_withoutDelta_chopped(feats, labels,indices, choplength, chopshift, minlength) :
    '''
    For creating chopped features for lstm training
    :param feats: input features as in input to DNNs (of size number_features x number_frames)
    :param labels: target features as in labels to DNNs (of size number_labels x number_frames)
    :param indices: indices denoting sentence boundaries in feats and labels
    :param choplength: maximum length for lstm training
    :param minlength : minimum length for lstm training
    :param chopshift: overlap fraction for chopping
    :return: feats_tensor, labels_tensor
    '''

    infeatdim = feats.shape[0]/2
    featlengths = indices[1,:] - indices[0,:] + 1
    numsamplesperutt = np.ceil((featlengths.astype(float) - minlength)/ chopshift).astype(int)

    feats_tensor = np.zeros((np.sum(numsamplesperutt), choplength, infeatdim))
    labels_tensor = np.zeros((np.sum(numsamplesperutt), choplength, labels.shape[0]))

    sindex = 0
    for k in range(indices.shape[1]) :
        secfeats = np.array(feats[:infeatdim  , indices[0,k] - 1 : indices[1,k]])
        seclabels = np.array(labels[:  , indices[0,k] - 1 : indices[1,k]])
        secfeats3d = np.zeros((numsamplesperutt[k], choplength, infeatdim))
        seclabels3d = np.zeros((numsamplesperutt[k], choplength, labels.shape[0]))
        startindex = 0
        for i in range(numsamplesperutt[k]) :
            endindex = startindex + choplength
            #print (startindex)
            #print (endindex)
            #print (secfeats.shape[1])
            if (endindex > secfeats.shape[1]):
                secf = np.concatenate((secfeats[:, startindex:], np.zeros((infeatdim, endindex - secfeats.shape[1]))), axis = 1)
                secl = np.concatenate((seclabels[:, startindex:], np.zeros((seclabels.shape[0], endindex - seclabels.shape[1]))), axis=1)
            else :
                secf = secfeats[:,startindex : endindex]
                secl = seclabels[:, startindex : endindex]
            secfeats3d[i,:,:] = secf.T
            seclabels3d[i,:,:] = secl.T
            startindex += chopshift

        eindex = sindex + numsamplesperutt[k]
        feats_tensor[sindex:eindex, : , : ] = secfeats3d
        labels_tensor[sindex:eindex, : , : ] = seclabels3d
        sindex = eindex
        #print(sindex)

    return feats_tensor, labels_tensor


def make_tensor_lstm_chopped(feats, labels,indices, choplength, chopshift, minlength) :
    '''
    For creating chopped features for lstm training
    :param feats: input features as in input to DNNs (of size number_features x number_frames)
    :param labels: target features as in labels to DNNs (of size number_labels x number_frames)
    :param indices: indices denoting sentence boundaries in feats and labels
    :param choplength: maximum length for lstm training
    :param minlength : minimum length for lstm training
    :param chopshift: overlap fraction for chopping
    :return: feats_tensor, labels_tensor
    '''

    infeatdim = feats.shape[0]
    featlengths = indices[1,:] - indices[0,:] + 1
    numsamplesperutt = np.ceil((featlengths.astype(float) - minlength)/ chopshift).astype(int)

    feats_tensor = np.zeros((np.sum(numsamplesperutt), choplength, infeatdim))
    labels_tensor = np.zeros((np.sum(numsamplesperutt), choplength, labels.shape[0]))

    sindex = 0
    for k in range(indices.shape[1]) :
        secfeats = np.array(feats[:infeatdim  , indices[0,k] - 1 : indices[1,k]])
        seclabels = np.array(labels[:  , indices[0,k] - 1 : indices[1,k]])
        secfeats3d = np.zeros((numsamplesperutt[k], choplength, infeatdim))
        seclabels3d = np.zeros((numsamplesperutt[k], choplength, labels.shape[0]))
        startindex = 0
        for i in range(numsamplesperutt[k]) :
            endindex = startindex + choplength
            #print (startindex)
            #print (endindex)
            #print (secfeats.shape[1])
            if (endindex > secfeats.shape[1]):
                secf = np.concatenate((secfeats[:, startindex:], np.zeros((infeatdim, endindex - secfeats.shape[1]))), axis = 1)
                secl = np.concatenate((seclabels[:, startindex:], np.zeros((seclabels.shape[0], endindex - seclabels.shape[1]))), axis=1)
            else :
                secf = secfeats[:,startindex : endindex]
                secl = seclabels[:, startindex : endindex]
            secfeats3d[i,:,:] = secf.T
            seclabels3d[i,:,:] = secl.T
            startindex += chopshift

        eindex = sindex + numsamplesperutt[k]
        feats_tensor[sindex:eindex, : , : ] = secfeats3d
        labels_tensor[sindex:eindex, : , : ] = seclabels3d
        sindex = eindex
        #print(sindex)

    return feats_tensor, labels_tensor



def make_tensor_lstm_chopped_reverse(feats, labels,indices, choplength, chopshift, minlength) :
    '''
    For creating chopped features for lstm training
    :param feats: input features as in input to DNNs (of size number_features x number_frames)
    :param labels: target features as in labels to DNNs (of size number_labels x number_frames)
    :param indices: indices denoting sentence boundaries in feats and labels
    :param choplength: maximum length for lstm training
    :param minlength : minimum length for lstm training
    :param chopshift: overlap fraction for chopping
    :return: feats_tensor, labels_tensor
    '''

    infeatdim = feats.shape[0]
    featlengths = indices[1,:] - indices[0,:] + 1
    numsamplesperutt = np.ceil((featlengths.astype(float) - minlength)/ chopshift).astype(int)

    feats_tensor = np.zeros((np.sum(numsamplesperutt), choplength, infeatdim))
    labels_tensor = np.zeros((np.sum(numsamplesperutt), choplength, labels.shape[0]))

    sindex = 0
    for k in range(indices.shape[1]) :
        secfeats = np.array(feats[:infeatdim  , indices[0,k] - 1 : indices[1,k]])
        seclabels = np.array(labels[:  , indices[0,k] - 1 : indices[1,k]])
        secfeats3d = np.zeros((numsamplesperutt[k], choplength, infeatdim))
        seclabels3d = np.zeros((numsamplesperutt[k], choplength, labels.shape[0]))
        startindex = 0
        for i in range(numsamplesperutt[k]) :
            endindex = startindex + choplength
            #print (startindex)
            #print (endindex)
            #print (secfeats.shape[1])
            if (endindex > secfeats.shape[1]):
                secf = np.concatenate((secfeats[:, startindex:], np.zeros((infeatdim, endindex - secfeats.shape[1]))), axis = 1)
                secl = np.concatenate((seclabels[:, startindex:], np.zeros((seclabels.shape[0], endindex - seclabels.shape[1]))), axis=1)
            else :
                secf = secfeats[:,startindex : endindex]
                secl = seclabels[:, startindex : endindex]
            secfeats3d[i,:,:] = np.fliplr(secf).T
            seclabels3d[i,:,:] = np.fliplr(secl).T
            startindex += chopshift

        eindex = sindex + numsamplesperutt[k]
        feats_tensor[sindex:eindex, : , : ] = secfeats3d
        labels_tensor[sindex:eindex, : , : ] = seclabels3d
        sindex = eindex
        #print(sindex)

    return feats_tensor, labels_tensor


def make_tensor_lstm_chopped_shift(feats, labels,indices, choplength, chopshift, minlength, shift) :
    '''
    For creating chopped features for lstm training
    :param feats: input features as in input to DNNs (of size number_features x number_frames)
    :param labels: target features as in labels to DNNs (of size number_labels x number_frames)
    :param indices: indices denoting sentence boundaries in feats and labels
    :param choplength: maximum length for lstm training
    :param minlength : minimum length for lstm training
    :param chopshift: overlap fraction for chopping
    :return: feats_tensor, labels_tensor
    '''

    infeatdim = feats.shape[0]
    featlengths = indices[1,:] - indices[0,:] + 1
    numsamplesperutt = np.ceil((featlengths.astype(float) - minlength)/ chopshift).astype(int)

    feats_tensor = np.zeros((np.sum(numsamplesperutt), choplength+ shift, infeatdim))
    labels_tensor = np.zeros((np.sum(numsamplesperutt), choplength+ shift, labels.shape[0]))

    sindex = 0
    for k in range(indices.shape[1]) :
        secfeats = np.array(feats[:infeatdim  , indices[0,k] - 1 : indices[1,k]])
        seclabels = np.array(labels[:  , indices[0,k] - 1 : indices[1,k]])
        secfeats3d = np.zeros((numsamplesperutt[k], choplength + shift, infeatdim))
        seclabels3d = np.zeros((numsamplesperutt[k], choplength + shift, labels.shape[0]))
        startindex = 0
        appendmatrixfeat = np.zeros((infeatdim, shift))
        prependmatrixlabel = np.zeros((seclabels.shape[0], shift))
        for i in range(numsamplesperutt[k]) :
            endindex = startindex + choplength
            #print (startindex)
            #print (endindex)
            #print (secfeats.shape[1])
            if (endindex > secfeats.shape[1]):
                secf = np.concatenate((secfeats[:, startindex:], np.zeros((infeatdim, endindex - secfeats.shape[1])), appendmatrixfeat), axis = 1)
                secl = np.concatenate((prependmatrixlabel, seclabels[:, startindex:], np.zeros((seclabels.shape[0], endindex - seclabels.shape[1]))), axis=1)
            else :
                secf = np.concatenate((secfeats[:,startindex : endindex], appendmatrixfeat),axis = 1)
                secl =np.concatenate((prependmatrixlabel, seclabels[:, startindex : endindex]),axis=1)
            secfeats3d[i,:,:] = np.fliplr(secf).T
            seclabels3d[i,:,:] = np.fliplr(secl).T
            startindex += chopshift

        eindex = sindex + numsamplesperutt[k]
        feats_tensor[sindex:eindex, : , : ] = secfeats3d
        labels_tensor[sindex:eindex, : , : ] = seclabels3d
        sindex = eindex
        #print(sindex)

    return feats_tensor, labels_tensor
