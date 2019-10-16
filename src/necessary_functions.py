"""
this function finds the files for the target images to create the dataset, which
is then used to create the dataloaders
"""

def get_y_fn(x):
    parent = 'train' if 'train' in str(x) else 'valid'
    fn = path_to_groundtruth/x.name
    #print(fn)
    #fn = data_dir/'flair'/parent/f'{str(x.stem)[:10]}-FLAIR_reg_zscore.nii.gz'
    return fn