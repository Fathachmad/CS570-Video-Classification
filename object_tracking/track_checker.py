import os, torch, pickle

for (root, dirs, files) in os.walk('../../data/data/'):
    for file in files:
        file_path = os.path.join(root, file)

        # Empty file : No track data
        if os.path.getsize(file_path) == 0:
            continue
        
        with open(file_path, 'rb') as pck:
            len_tensor = pickle.load(pck)
            track_tensor = pickle.load(pck)

            if not torch.is_tensor(len_tensor) or not torch.is_tensor(track_tensor):
                print('{} is not tensor'.format(file_path))
                continue
            
            if len_tensor.size(0) != track_tensor.size(0):
                print('{} has incorrect track/length data'.format(file_path))
                continue
            
            if track_tensor.size(2) != 6:
                print('{} has not enough track data'.format(file_path))
                continue