from random import randint

import numpy as np
import pylab as pl
from scikits.learn.decomposition import RandomizedPCA

from drawing import draw_2d

def main():
    dataset = []

    # create a random dataset with points on the X=Y axis
    for i in range(100):
        for n in range(randint(1, 10)):
            dataset.append((i+randint(-5,+5), i+randint(-5,+5)))

    dataset = np.array(dataset)
    draw(dataset, 'before.png')

    # run a PCA to 2 dimensions for this dataset
    transformed_dataset = RandomizedPCA(2).fit(dataset).transform(dataset)
    from ipdb import set_trace; set_trace()

    draw_2d(transformed_dataset, 'after.png')

if __name__ == '__main__':
    main()
