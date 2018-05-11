import pydensecrf.densecrf as dcrf
import numpy as np

def dense_crf(probs,n_classes ,img=None, n_iters=10,
             sxy_gaussian=(3, 3), compat_gaussian=3,
             kernel_gaussian=dcrf.DIAG_KERNEL,
             normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
             sxy_bilateral=(49, 49), compat_bilateral=4,
             srgb_bilateral=(5, 5, 5),
             kernel_bilateral=dcrf.DIAG_KERNEL,
             normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):

   """DenseCRF over unnormalised predictions.
      More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

   Args:
  probs: class probabilities per pixel.
  img: if given, the pairwise bilateral potential on raw RGB values will be computed.
  n_iters: number of iterations of MAP inference.
  sxy_gaussian: standard deviations for the location component of the colour-independent term.
  compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
  kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FU
  LL_KERNEL).
  normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEF
  ORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
  sxy_bilateral: standard deviations for the location component of the colour-dependent term.
  compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
  srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
  kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FUL
  L_KERNEL).
  normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFO
  RE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

  Returns:
  Refined predictions after MAP inference.
  """
   _, _, h, w = probs.shape

   img = np.expand_dims(img,0).copy(order='C')
   #print img.flags
   probs = probs[0].transpose(0, 1, 2).copy(order='C')
   #print probs.flags
   #probs = np.ascontiguousarray(probs)
   #n_classes = 21
   #probs = probs[0].transpose(2, 0, 1).copy(order='C')  # Need a contiguous array.

   d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
   U = -np.log(probs)  # Unary potential.
   U = U.reshape((n_classes, -1))  # Needs to be flat.
   d.setUnaryEnergy(U)
   d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                      kernel=kernel_gaussian, normalization=normalisation_gaussian)
   if img is not None:
       assert (img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
       d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                           kernel=kernel_bilateral, normalization=normalisation_bilateral,
                           srgb=srgb_bilateral, rgbim=img[0])
   Q = d.inference(n_iters)
   preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w))
   return np.expand_dims(preds, 0)

def pred_union(output1,output2):

    assert output1.shape == output2.shape
    mask = output1 + output2
    mask_union = mask.copy()
    mask_union[mask_union == 2]=1

    return mask_union

def pred_insect(output1,output2):

    assert output1.shape == output2.shape
    mask = output1 - output2
    mask_insect = mask.copy()
    mask_insect[mask_insect == 1]= 0
    mask_insect[mask_insect == -1] = 0
    mask_insect[mask_insect == 0] = 1

    return mask_insect