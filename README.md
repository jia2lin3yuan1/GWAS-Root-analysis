#GWAS-Root-analysis

### Train two segmentation models:
	1. Rotate images to ensure: plant is placed as leaf on top, root on bottom. Ruler and the labels are on the bottom-right corner. It helps to simplify the input patterns for the network to learn.
	2. Segment BG based on color intensity, and then segment the plant, ruler, and label as three isolated connected regions based on their spatial relationship.
	3. Collect good examples, and train a deeplab-resnet50 model to segment image into BG, plant, Ruler, and Label
	
	4. Based on step2's result, zoom in to the local window containing plant on the RGB image, segment the plant into Leaf, Stem, and Root based on color intensity.
	5. Collect good examples and train another deeplabe-resnet50 model to segment the image into BG, Leaf, Stem, and Root.
	
### In inference stage:
	1. Rotate images same as the training stage. 
	2. Segment the image to plant, ruler, and label with the 1st segmentation model
	3. Perform same ZoomIn as in the trianing stage, zoomin to the local window of plant on RGB image and get its segmentation result from the 2nd segmentaion model, refine the segmentation of leaf, stem, and root. 
	4. Compute the height of the ruler. Because its physical heigh is fixed, so we get the size of each pixel
	5. Compute the leaf area, stem width, stem area
	6. Compute root traits:
		+ Isolated the segment of root into individuals
		+ For each single root, classify its pixels as major root or minor root based on the geometric analysis
		+ For each single root, classify as basal or lateral roots based on if they are connected to the upper BG or bottom BG

        + For each single root, compute its traits: length, area, major area, minor area, number of minor roots, root type as basal or lateral, 


## Running script:

+ training pipeline
    * uniform rotation: 
        ```bash
        python step1_preprocess_rotation.py
        ```
    * train network for segmentation of plant / ruler / label
    * run deep-model-1 and zoom-in to crop image
        ```bash
        python step2_segment_ms.py
        ```
    * train network for segmentation of leaf / stem /root

+ configuration is setted in 'config.py'
+ run inference pipeline on new input:
    ```bash
    python step1_preprocess_rotation.py
    python step2_segment_ms.py
    python step3_compute_traits.py
    ```
    
## Acknowledgements

We thank the National Science Foundation Plant Genome Research Program for support (IOS #1546900, Analysis of genes affecting plant regeneration and transformation in poplar), and members of GREAT TREES Research Cooperative at OSU for its support of the Strauss laboratory. 
Support for the Poplar GWAS dataset is provided by the U.S. Department of Energy, Office of Science Biological and Environmental Research (BER) via the Center for Bioenergy Innovation (CBI) under Contract No. DE-PS02-06ER64304. The Poplar GWAS Project used resources of the Oak Ridge Leadership Computing Facility and the Compute and Data Environment for Science at Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725. We would like to thank the efforts of personnel from the CBI in establishing the GWAS resource used for this study.

