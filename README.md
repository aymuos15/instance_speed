brats.py -> Original Version from: https://github.com/rachitsaluja/BraTS-2023-Metrics

brats_optimised -> Soumya's version to speed up the code.

soumya.py -> GPU/Torch version of brats_optimised.

The test.py contains the comparison of all the methods.
The panoptica (https://github.com/BrainLesion/panoptica) functions are based on two matches. 
Naive is one to one and the other is many to one.

TODO:
- GPU Version on Connected Components
- Understand why and where the difference in scores occur between panoptica and the brats scores. There is no difference in the case of only one instance.


NOTES:
- Had to make my own fork of panoptica for some minor changes to get the different matchings working and remove redundant prints.
- Currently, CPU is faster than GPU. I am not sure why.

DATA:
Our Inhouse Mets-data with marina's nnunet results.
