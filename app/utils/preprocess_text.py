# This function saves the input array X to a .npy file with the given path/filename
def save_X_to_npy(X, path):
    np.save(path, arr= X)
