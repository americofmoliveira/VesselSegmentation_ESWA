# Retinal vessel segmentation based on Fully Convolutional Neural Networks

This repository contains materials supporting the paper: Oliveira, Américo, et al., "Retinal vessel segmentation based on Fully Convolutional Neural Networks", Expert Systems with Applications, Volume --, -- 2018. [[link]()]

## Abstract

The retinal vascular condition is a reliable biomarker of several ophthalmologic and cardiovascular diseases, so automatic vessel segmentation may be crucial to diagnose and monitor them. In this paper, we propose a novel method that combines the multiscale analysis provided by the Stationary Wavelet Transform with a multiscale Fully Convolutional Neural Network to cope with the varying width and direction of the vessel structure in the retina. Our proposal uses rotation operations as the basis of a joint strategy for both data augmentation and prediction, which allows us to explore the information learned during training to refine the segmentation. The method was evaluated on three publicly available databases, achieving an average accuracy of 0.9576, 0.9694, and 0.9653, and average area under the ROC curve of 0.9821, 0.9905, and 0.9855 on the DRIVE, STARE, and CHASE_DB1 databases, respectively. It also appears to be robust to the training set and to the inter-rater variability, which shows its potential for real-world applications.

## Contents

The materials in this repository are organized as follows:

- `code`: Code required to test our model. (*In Progress*)

- `folds_constitution`: Since there is no explicit division between training and test sets for the STARE and CHASE_DB1 databases, we used *5*-fold and *4*-fold cross-validation, respectively, to evaluate the results in these cases. Here we show the constitution of the folds, so that future works can replicate our evaluation conditions.

- `image_level_results`: Evaluation metrics for each image in terms of Sensitivity, Specificity, Accuracy, Area under the ROC curve (AUC), and Matthews correlation coefficient (MCC). For STARE, in particular, we also provide the performance of our model in the set of pathological images.

- `resources`: Mask, probability map outputted by the model, and final binary segmentation for each image. 

- `statistical_comparison`: Statistical comparison between our method and other state-of-the-art works that have made their segmentations publicly available.

## Contact

For information related with the paper, please feel free to contact me (americofmoliveira@gmail.com) or Prof. Carlos A. Silva (csilva@dei.uminho.pt).
