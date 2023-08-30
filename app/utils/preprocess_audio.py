# This function saves the input array y to a .npy file with the given path/filename
def save_y_to_npy(y, path):
    np.save(path, arr= y)
