#https://github.com/usamafayaz/MachineLearning.git
# Accuracy and Speed
# 1. #
# Ways to Increase Accuracy:
# 1. Adding some hidden layers.
# 2. Adding number of neurons in hidden layers.
# 3. Adding more images per class.
# 4. Data Augmentation.
#       Add Salt and Pepper noise in 20k images.
#       Add 20k noisy images.
#       Add 20k scaling
#       Add 20k blue filter images
#       Add 20k rotated images

# 5. OverFitting => More Accuracy in Training but less Accuracy in Testing.
# Solution of OverFitting
#   1. Early Stop.
        # Stop Training at point where gap between training and testing is minimum and loss is also less.
#   2. Drop Out.
        # Stop some neurons for some time. Kick Out brilliant students for some time.

# 6. DataSet Imbalance
    # A-100 || B-20 || C-40
    # A is majority class and B is minority class
    # Solution of DataSet Imbalance:
        # 1. OverSampling => increase in Minority class.
        # 2. UnderSampling => SMOTE algorithm.

###################################################################

# 2. #
# Speed and Real Time Performance.

###################################################################

# Feature Extraction
# Neural Network always works on number.

# Features => Defining Characteristics through which we could identify difference between one object from another.

# Hand Crafted 2012-2013 || Humans took out the Features.
# 2013 Concolution Neural Network (CNN) is revolutionary thing !

# Concolution Neural Network (CNN)

# 128x128  ==>  Conv Net 64 ==> 126x126x64 ==> Max Pooling ==> 63x63x64 ==> Conv Net 128 ==> 61x61x128 ==> Max Pooling ==> 30x30x128 ==> Conv Net 128 ==> 28x28x128 ==> Max Pooling ==> 14x14x128
