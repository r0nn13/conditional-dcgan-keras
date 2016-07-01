## Bugs ##
There are a few bugs in the code, and convergence issues that I hope to fix soon the moment I get access to a decent GPU / have some more time. Please do not waste hours training the model as is. If you get a chance to fix anything, please make a PR.


## KERAS-DCGAN ##
Implementation of http://arxiv.org/abs/1511.06434 with the (awesome) [keras](https://github.com/fchollet/keras) library, for generating artificial images with deep learning.

This trains two adversarial deep learning models on real images, in order to produce artificial images that look real.

The generator model tries to produce images that look real and get a high score from the discriminator.

The discriminator model tries to tell apart between real images and artificial images from the generator.

Usage
-----
**Training:**
 `python dcgan.py --mode train --path <path_to_images> --batch_size <batch_size>`
    
  python dcgan.py --mode train --path ~/images --batch_size 128

**Image generation:**
`python dcgan.py --mode generate --batch_size <batch_size>`

python dcgan.py --mode generate --batch_size 128

