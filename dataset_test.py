from expDataset import expDatasetSynthetic

def test_toydataset():
    ds = expDatasetSynthetic(100,10,600,0.8,0.4,beta=0.2,seed=1991)
    X = ds.load()
    assert X.shape == (100,600)
