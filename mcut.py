import numpy as np
import nibabel as nib
import pickle
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from bunch import Bunch

def savepkl(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        output.close()
        
data = datasets.fetch_adhd(n_subjects=40)
atlas = datasets.fetch_atlas_aal()
atlas_filename = atlas.maps
atlas_labels = atlas.labels

adhd_ind=[]
tdc_ind=[]
for i in xrange(40):
    if data.phenotypic[i]["adhd"]==1:
        adhd_ind.append(i)
    else:
        tdc_ind.append(i)
        
data_tdc = list(np.array(data.func)[tdc_ind])
data_tdc_confunds = list(np.array(data.confounds)[tdc_ind])
data_adhd = list(np.array(data.func)[adhd_ind])
data_adhd_confunds = list(np.array(data.confounds)[adhd_ind])

aal_data =nib.load(atlas_filename).get_data()
aal_affine=nib.load(atlas_filename).get_affine()

full_mask = np.ones((91,109,91)).astype(int)
full_mask = nib.nifti1.Nifti1Image(full_mask,aal_affine)
full_masker = NiftiMasker(mask_img=full_mask,mask_strategy="epi",
                               memory='nilearn_cache', 
                               memory_level=1,standardize=False)

ts = [full_masker.fit_transform(data_tdc[i], 
                                confounds=data_tdc_confunds[i]) 
      for i in xrange(20)]

tdc_masked = np.vstack(ts)
tdc_masked_4d = tdc_masked.T.reshape(91,109,91,-1)

ts = [full_masker.fit_transform(data_adhd[i], 
                                confounds=data_adhd_confunds[i]) 
      for i in xrange(20)]

adhd_masked = np.vstack(ts)
adhd_masked_4d = adhd_masked.T.reshape(91,109,91,-1)

cut = 45
slice_mask = (aal_data[:,:,cut]!=0)
slice_mask_flatten = np.array(slice_mask.flatten())
length = len(slice_mask_flatten)

tdc_cut = tdc_masked_4d[:,:,cut,:].reshape(length,-1)[slice_mask_flatten,:]
adhd_cut = adhd_masked_4d[:,:,cut,:].reshape(length,-1)[slice_mask_flatten,:]

mcut = Bunch()
mcut.tdc_cut = tdc_cut
mcut.adhd_cut = adhd_cut
mcut.cut = cut 
mcut.slice_mask_flatten = slice_mask_flatten
mcut.slice_mask = slice_mask

savepkl(mcut,"mcut.pkl")