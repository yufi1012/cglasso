from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker,NiftiMasker
import nibabel as nib
import numpy as np
from sklearn.feature_extraction import image
import pickle
from bunch import Bunch
######################################################################
def savepkl(obj,filename):
    with open(filename,'wb') as output:
        pickle.dump(obj,output,pickle.HIGHEST_PROTOCOL)
        output.close()

def loadpkl(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)
        f.close()
#######################################################################

data = datasets.fetcj_adhd(n_subjects=40)
atlas = datasets.fetch_atlas_aal()
atlas_filename = atlas.maps
atlas_labels = atlas.labels

adhd_ind = []
tdc_ind = []
for i in xrange(40):
    if data.phenotypic[i][22]==1:
        adhd_ind.append(i)
    else:
        tdc_ind.append(i)

data_tdc = list(np.array(data.func)[tdc_ind])
data_tdc_confunds = list(np.array(data.confounds)[tdc_ind])
data_adhd = list(np.array(data.func)[adhd_ind])
data_adhd_confunds = list(np.array(data.confounds)[adhd_ind])

test_data = nib.load(atlas_filename).get_data()
test_affine = nib.load(atlas_filename).get_affine()

aal_mask = test_data.astupe(bool).astype(int)
aal_mask_obj = nib.nifti1.Nifti1Image(aal_mask,test_affine)
aal_nifti_masker = NiftiMasker(mask_img=aal_mask_obj,mask_strategy="epi",
                               memory='nilearn_cache',
                               memory_level=1,standardize=False)

ts = [aal_nifti_masker.fit_transform(data_tdc[i], 
                                confounds=data_tdc_confunds[i]) 
      for i in xrange(20)]
tdc_masked = np.vstack(ts)
ts = [aal_nifti_masker.fit_transform(data_adhd[i], 
                                confounds=data_adhd_confunds[i]) 
      for i in xrange(20)]
adhd_masked = np.vstack(ts)

shape = aal_mask.shape
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=aal_mask.astype(bool))


from sklearn.cluster import FeatureAgglomeration
# If you have scikit-learn older than 0.14, you need to import
# WardAgglomeration instead of FeatureAgglomeration
import time
start = time.time()
ward = FeatureAgglomeration(n_clusters=2000, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
ward.fit(tdc_masked)
print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

tdc_labels = ward.labels_ + 1
tdc_labels_img = aal_nifti_masker.inverse_transform(tdc_labels)
tdc_reduced = ward.transform(tdc_masked)


start = time.time()
ward.fit(adhd_masked)
print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

adhd_labels = ward.labels_ + 1
adhd_labels_img = aal_nifti_masker.inverse_transform(adhd_labels)
adhd_reduced = ward.transform(adhd_masked)

output = Bunch()
output.tdc_labels = tdc_labels
output.tdc_labels_img = tdc_labels_img
output.tdc_reduced = tdc_reduced

output.adhd_labels = adhd_labels
output.adhd_labels_img = adhd_labels_img
output.adhd_reduced = adhd_reduced

savepkl(output,"adhd40_reduced.pkl")


