# UltraNet

Abstract—Ultrasound image segmentation is an essential tool for recognizing different parts of an organ and performing clinical diagnoses. However, in ultrasound image segmentation with high uncertainty, satisfactory segmentation results are difficult to achieve because of the highly ambiguous nature of the boundaries. Moreover, similar objects with similar pixel values result in frequent misidentification. Thus, this paper focuses on using a new boundary preservation module and class context module to solve these problems in ultrasound image segmentation with high uncertainty with a novel coarse-to-fine network called Ultranet. The proposed dynamic boundary preservation module uses a dynamic key boundary point map to enhance the boundary module, and the coarse label mask from one stage is used to obtain a multiscale class-context filter to improve the segmentation performance. Finally, two datasets, Neck Muscle and Thyroid, are used to demonstrate that our proposed Ultranet outperforms other state-of-the art methods, such as CE-Net and MSSGAnet, in terms of the mean intersection-over-union (MIOU), Dice coefficient and mean pixel accuracy (MPA), e.g., obtaining 0.823 MIOU on the Thyroid dataset and 0.840 MIOU on the Neck Muscle dataset.

# The project code will upload once the paper acctept!

The current version is an unorganized draft, which can be used to train DBPB model. 

you can train the model by run:

  # python main.py
  
and test model by run:

  # python predict_supersound.py
