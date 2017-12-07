import os
def get_files(datadir):
    files = [os.path.join(datadir, x) for x in os.listdir(datadir)]
    print files
    return files