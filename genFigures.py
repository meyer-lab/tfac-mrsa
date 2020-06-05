#!/usr/bin/env python3

import sys
import logging
import time
import matplotlib
matplotlib.use('AGG')

fdir = './output/'

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if __name__ == '__main__':
    start = time.time()
    nameOut = 'figure' + sys.argv[1]

    exec('from tfac.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=300, bbox_inches='tight', pad_inches=0)

    logging.info(f'Figure {sys.argv[1]} is done after {time.time() - start} seconds.')
