# Retinal vessel segmentation based on Fully Convolutional Neural Networks

This repository contains materials supporting the paper: Oliveira, Américo, et al., "Retinal vessel segmentation based on Fully Convolutional Neural Networks", Expert Systems with Applications, Volume --, -- 2018. [[link](https://www.sciencedirect.com/science/article/pii/S0957417418303816)]

## Abstract

The retinal vascular condition is a reliable biomarker of several ophthalmologic and cardiovascular diseases, so automatic vessel segmentation may be crucial to diagnose and monitor them. In this paper, we propose a novel method that combines the multiscale analysis provided by the Stationary Wavelet Transform (SWT) with a multiscale Fully Convolutional Neural Network (FCN) to cope with the varying width and direction of the vessel structure in the retina. Our proposal uses rotation operations as the basis of a joint strategy for both data augmentation and prediction, which allows us to explore the information learned during training to refine the segmentation. The method was evaluated on three publicly available databases, achieving an average accuracy of 0.9576, 0.9694, and 0.9653, and average area under the ROC curve of 0.9821, 0.9905, and 0.9855 on the DRIVE, STARE, and CHASE_DB1 databases, respectively. It also appears to be robust to the training set and to the inter-rater variability, which shows its potential for real-world applications.

## Overview

![Pipeline](https://github.com/americofmoliveira/VesselSegmentation_ESWA/blob/master/resources/architecture/1a.png)

![Architecture](https://github.com/americofmoliveira/VesselSegmentation_ESWA/blob/master/resources/architecture/1b.png)

## Comparison with the state-of-the-art

<table class="tg">
  <tr>
    <th class="tg-lm6i" rowspan="2"><br>Method (year)</th>
    <th class="tg-lm6i" colspan="4">DRIVE</th>
    <th class="tg-lm6i" colspan="4">STARE</th>
    <th class="tg-lm6i" colspan="4">CHASE_DB1</th>
  </tr>
  <tr>
    <td class="tg-lm6i">Sn</td>
    <td class="tg-lm6i">Sp</td>
    <td class="tg-lm6i">Acc</td>
    <td class="tg-lm6i">AUC</td>
    <td class="tg-lm6i">Sn</td>
    <td class="tg-lm6i">Sp</td>
    <td class="tg-lm6i">Acc</td>
    <td class="tg-lm6i">AUC</td>
    <td class="tg-lm6i">Sn</td>
    <td class="tg-lm6i">Sp</td>
    <td class="tg-lm6i">Acc</td>
    <td class="tg-lm6i">AUC</td>
  </tr>
  <tr>
    <td class="tg-7x02">Mendonça and Campilho (2006)</td>
    <td class="tg-akyt">0.7344</td>
    <td class="tg-akyt">0.9764</td>
    <td class="tg-akyt">0.9452</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-akyt">0.6996</td>
    <td class="tg-akyt">0.9730</td>
    <td class="tg-akyt">0.9440</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
  </tr>
  <tr>
    <td class="tg-7x02">Soares <span style="font-style:italic">et al.</span> (2006)</td>
    <td class="tg-akyt">0.7332</td>
    <td class="tg-akyt">0.9782</td>
    <td class="tg-akyt">0.9466</td>
    <td class="tg-akyt">0.9614</td>
    <td class="tg-akyt">0.7207</td>
    <td class="tg-akyt">0.9747</td>
    <td class="tg-akyt">0.9480</td>
    <td class="tg-akyt">0.9671</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
  </tr>
  <tr>
    <td class="tg-7x02">Fraz <span style="font-style:italic">et al.</span> (2012)</td>
    <td class="tg-akyt">0.7406</td>
    <td class="tg-akyt">0.9807</td>
    <td class="tg-akyt">0.9480</td>
    <td class="tg-akyt">0.9747</td>
    <td class="tg-akyt">0.7548</td>
    <td class="tg-akyt">0.9763</td>
    <td class="tg-akyt">0.9534</td>
    <td class="tg-akyt">0.9768</td>
    <td class="tg-akyt">0.7224</td>
    <td class="tg-akyt">0.9711</td>
    <td class="tg-akyt">0.9469</td>
    <td class="tg-akyt">0.9712</td>
  </tr>
  <tr>
    <td class="tg-7x02">Azzopardi <span style="font-style:italic">et al.</span> (2015)</td>
    <td class="tg-akyt">0.7655</td>
    <td class="tg-akyt">0.9704</td>
    <td class="tg-akyt">0.9442</td>
    <td class="tg-akyt">0.9614</td>
    <td class="tg-akyt">0.7716</td>
    <td class="tg-akyt">0.9701</td>
    <td class="tg-akyt">0.9497</td>
    <td class="tg-akyt">0.9563</td>
    <td class="tg-akyt">0.7585</td>
    <td class="tg-akyt">0.9587</td>
    <td class="tg-akyt">0.9387</td>
    <td class="tg-akyt">0.9487</td>
  </tr>
  <tr>
    <td class="tg-7x02">Roychowdhury <span style="font-style:italic">et al.</span> (2015)</td>
    <td class="tg-akyt">0.7249</td>
    <td class="tg-qpkk">0.9830</td>
    <td class="tg-akyt">0.9519</td>
    <td class="tg-akyt">0.9620</td>
    <td class="tg-akyt">0.7719</td>
    <td class="tg-akyt">0.9726</td>
    <td class="tg-akyt">0.9515</td>
    <td class="tg-akyt">0.9688</td>
    <td class="tg-akyt">0.7201</td>
    <td class="tg-akyt">0.9824</td>
    <td class="tg-akyt">0.9530</td>
    <td class="tg-akyt">0.9532</td>
  </tr>
  <tr>
    <td class="tg-7x02">Li <span style="font-style:italic">et al.</span> (2016)</td>
    <td class="tg-akyt">0.7569</td>
    <td class="tg-akyt">0.9816</td>
    <td class="tg-akyt">0.9527</td>
    <td class="tg-akyt">0.9738</td>
    <td class="tg-akyt">0.7726</td>
    <td class="tg-akyt">0.9844</td>
    <td class="tg-akyt">0.9628</td>
    <td class="tg-akyt">0.9879</td>
    <td class="tg-akyt">0.7507</td>
    <td class="tg-akyt">0.9793</td>
    <td class="tg-akyt">0.9581</td>
    <td class="tg-akyt">0.9716</td>
  </tr>
  <tr>
    <td class="tg-7x02">Zhang <span style="font-style:italic">et al.</span> (2016)</td>
    <td class="tg-akyt">0.7743</td>
    <td class="tg-akyt">0.9725</td>
    <td class="tg-akyt">0.9476</td>
    <td class="tg-akyt">0.9636</td>
    <td class="tg-akyt">0.7791</td>
    <td class="tg-akyt">0.9758</td>
    <td class="tg-akyt">0.9554</td>
    <td class="tg-akyt">0.9748</td>
    <td class="tg-akyt">0.7626</td>
    <td class="tg-akyt">0.9661</td>
    <td class="tg-akyt">0.9452</td>
    <td class="tg-akyt">0.9606</td>
  </tr>
  <tr>
    <td class="tg-7x02">Liskowski and Krawiec (2016)</td>
    <td class="tg-akyt">0.7520</td>
    <td class="tg-akyt">0.9806</td>
    <td class="tg-akyt">0.9515</td>
    <td class="tg-akyt">0.9710</td>
    <td class="tg-akyt">0.8145</td>
    <td class="tg-qpkk">0.9866</td>
    <td class="tg-qpkk">0.9696</td>
    <td class="tg-akyt">0.9880</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
    <td class="tg-gzo9">-</td>
  </tr>
  <tr>
    <td class="tg-7x02">Orlando <span style="font-style:italic">et al.</span> (2017)</td>
    <td class="tg-akyt">0.7897</td>
    <td class="tg-akyt">0.9684</td>
    <td class="tg-akyt">0.9454</td>
    <td class="tg-akyt">0.9506</td>
    <td class="tg-akyt">0.7680</td>
    <td class="tg-akyt">0.9738</td>
    <td class="tg-akyt">0.9519</td>
    <td class="tg-akyt">0.9570</td>
    <td class="tg-akyt">0.7565</td>
    <td class="tg-akyt">0.9655</td>
    <td class="tg-akyt">0.9467</td>
    <td class="tg-akyt">0.9478</td>
  </tr>
  <tr>
    <td class="tg-7x02">Zhang <span style="font-style:italic">et al.</span> (2017)</td>
    <td class="tg-akyt">0.7861</td>
    <td class="tg-akyt">0.9712</td>
    <td class="tg-akyt">0.9466</td>
    <td class="tg-akyt">0.9703</td>
    <td class="tg-akyt">0.7882</td>
    <td class="tg-akyt">0.9729</td>
    <td class="tg-akyt">0.9547</td>
    <td class="tg-akyt">0.9740</td>
    <td class="tg-akyt">0.7644</td>
    <td class="tg-akyt">0.9716</td>
    <td class="tg-akyt">0.9502</td>
    <td class="tg-akyt">0.9706</td>
  </tr>
  <tr>
    <td class="tg-7x02">This work (2018)</td>
    <td class="tg-qpkk">0.8039</td>
    <td class="tg-akyt">0.9804</td>
    <td class="tg-qpkk">0.9576</td>
    <td class="tg-qpkk">0.9821</td>
    <td class="tg-qpkk">0.8315</td>
    <td class="tg-akyt">0.9858</td>
    <td class="tg-akyt">0.9694</td>
    <td class="tg-qpkk">0.9905</td>
    <td class="tg-qpkk">0.7779</td>
    <td class="tg-qpkk">0.9864</td>
    <td class="tg-qpkk">0.9653</td>
    <td class="tg-qpkk">0.9855</td>
  </tr>
</table>

## Contents

The materials in this repository are organized as follows:

- `code`: Code required to test our model. (*In Progress*)

- `folds_constitution`: Since there is no explicit division between training and test sets for the STARE and CHASE_DB1 databases, we used *5*-fold and *4*-fold cross-validation, respectively, to evaluate the results in these cases. Here we show the constitution of the folds, so that future works can replicate our evaluation conditions.

- `image_level_results`: Evaluation metrics for each image in terms of Sensitivity, Specificity, Accuracy, Area under the ROC curve (AUC), and Matthews correlation coefficient (MCC). For STARE, in particular, we also provide the performance of our model in the set of pathological images.

- `resources`: Mask, probability map outputted by the model, and final binary segmentation for each image. 

- `statistical_comparison`: Statistical comparison between our method and other state-of-the-art works that have made their segmentations publicly available.

## Contact

For information related with the paper, please feel free to contact me (americofmoliveira@gmail.com) or Prof. Carlos A. Silva (csilva@dei.uminho.pt).
