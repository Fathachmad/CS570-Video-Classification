from yololstm import YoloLstm

models = ['birds_planes_200e', 'full_dataset_100e', 'detect_best']
augmented = [False, True]

for model in models:
    print('Working on model {}'.format(model))

    for augment in augmented:
        print('{}ugmented dataset'.format('A' if augment else 'Not a'))
        a = YoloLstm(model, augment)
        a.train(200)