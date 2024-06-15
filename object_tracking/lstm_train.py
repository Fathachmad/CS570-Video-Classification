from yololstm import YoloLstm

models = ['full_ft', 'full_augmented_ft', 'full_dataset_100e.pt']
augmented = [False, True]

for model in models:
    print('Working on model {}'.format(model))

    for augment in augmented:
        print('{}ugmented dataset'.format('A' if augment else 'Not a'))
        a = YoloLstm(model, augment)
        a.train(500)
