# Neural Representation Benchmark

See http://dicarlolab.mit.edu/neuralbenchmark for more information.

### For help:
python commandline_interface.py --help

    usage: commandline_interface.py [-h] [--image_dir IMAGE_DIR]
                                    [--feature_dir FEATURE_DIR]
    
    optional arguments:
      -h, --help            show this help message and exit
      --image_dir IMAGE_DIR
                            Path to NeuralRepresentationImages_7x7 directory, for example:
                                /Users/cadieu/Downloads/NeuralRepresentationImages_7x7_20130408
      --feature_dir FEATURE_DIR
                            Path to feature directory containing files for each variation level.
                            These files should be named: Variation00_20110203.txt,
                                                         Variation03_20110128.txt,
                                                         Variation06_20110131.txt
                            Each line in the file is the feature for an image, beginning with the image id:
                                Variation00_20110203/b0c7d9523215b272249d84287e1d28d851275f4f.png -0.45 -0.74 -0.37
                                Variation00_20110203/d0f6824ddcaf80456ceb0e86def346fa2ac97112.png -0.28 -0.31 -0.27


### Notes:
The commandline_interface.py expects the features on disk in .txt format.
If your features are large, or you want to avoid the dump to text files, 
you can format appropriately in python/numpy and call the appropriate functions directly.
