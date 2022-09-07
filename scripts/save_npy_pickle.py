from scripts import utils
import glob
import pickle

vel_data = sorted(glob.glob('../Dataset/*.dat'))
# saving npy
for datafile in vel_data:
    idx2 = int(datafile[21:-14]) # number from a filename !!!!!!!!
    f = np.loadtxt(datafile)
    f = np.reshape(f,(256,256,2))
    f = np.rot90(f)
    CG_f_0 = utils.coarse_grain(f[:,:,0],2)/(2*2)
    CG_f_1 = utils.coarse_grain(f[:,:,1],2)/(2*2)
    with open(f'../Dataset2/numpysave_{idx2}.npy','wb') as File:
        np.save(File, f)
        np.save(File, CG_f_0)
        np.save(File, CG_f_1)

# saving pickle
for datafile in vel_data:
    idx2 = int(datafile[21:-14]) # number from a filename !!!!!!!!
    f = np.loadtxt(datafile)
    f = np.reshape(f,(256,256,2))
    f = np.rot90(f)
    CG_f_0 = utils.coarse_grain(f[:,:,0],2)/(2*2)
    CG_f_1 = utils.coarse_grain(f[:,:,1],2)/(2*2)
    with open(f'../Dataset3/picklesave_{idx2}','wb') as File:
        pickle.dump(f, File)
        pickle.dump(CG_f_0, File)
        pickle.dump(CG_f_1, File)

# chacking that saving was a success
for datafile in vel_data:
    idx2 = int(datafile[21:-14])
    f = np.loadtxt(datafile)
    f = np.reshape(f,(256,256,2))
    f = np.rot90(f)
    CG_f_0 = utils.coarse_grain(f[:,:,0],2)/(2*2)
    CG_f_1 = utils.coarse_grain(f[:,:,1],2)/(2*2)
    with open(f'../Dataset2/numpysave_{idx2}.npy','rb') as File:
        f1 = np.load(File)
        CG_f_01 = np.load(File)
        CG_f_11 = np.load(File)
    with open(f'../Dataset3/picklesave_{idx2}','rb') as File:
        f2 = pickle.load(File)
        CG_f_02 = pickle.load(File)
        CG_f_12 = pickle.load(File)
    assert (f==f1).all() and (f==f2).all()
    assert (CG_f_0==CG_f_01).all() and (CG_f_0==CG_f_02).all()
    assert (CG_f_1==CG_f_11).all() and (CG_f_1==CG_f_12).all()