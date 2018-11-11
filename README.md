Clothes Recommendation
======================

Customers are often interested in see products similar to the ones
they have bought in the past. This clothes recommender ranks images
according to their similarity to a given image. A clothes dataset used
in this demonstration contains images with three labels: dotted,
leopard, and striped. The goal is to build a recommendation model to
recommend new clothes images to users.

The recommendation model uses CNN for image featurization and SVM for
similarity ranking, aiming for a more explanable knowledge
representation. Computation time necessitates a GPU based DLVM.

Visit the github repository for more details of this version of the model:
<https://github.com/gjwgit/clothes>

Usage
-----

The **demo** command applies the pre-built model to a demo data set
with around 60 images and shows the top-1 recommendation results.

The **print** command displays a textual summary of the pre-trained
CNN model and its build parameters.

The **display** will display a visual representation of the model.

Run the **score** command to provide recommendation for new clothes
texture data.

The following link provides a deep guidance on how the model is built
using Microsoft Cognitive Toolkit (CNTK).
\[&lt;<https://github.com/Azure/ImageSimilarityUsingCntk>&gt;\]
