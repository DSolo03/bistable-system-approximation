import pickle

def saveDataset(dataset, filename:str = "dataset.pickle"):
    with open(f'data/{filename}', 'wb') as f:
        pickle.dump(dataset, f)
        
def loadDataset(filename:str = "dataset.pickle"):
    with open(f'data/{filename}', 'rb') as f:
        return pickle.load(f)