# Rager
Rager (Rocket for Action and GEsture Recognition) is a modified MultiRocket transformer for human action and gesture recognition based on skeletal data.

# How does it work?
Rager introduces mechanics such as Zero-Sum Dipole and mean absolute deviation for alternative channel aggregation, as well as a redefined set of features with a dual cut-off threshold system. The goal of these modifications was to increase accuracy and reduce training and classification times compared to Rocket methods on HAGR data while retaining all of their previously mentioned advantages.

# Languages
Rager is available in Python.

# How to use it?
The Rager implementation is in the file "transformations\collection\convolution_based\_rager.py"

The implementation is in the aeon 1.4.0 module format (https://www.aeon-toolkit.org). To use it, copy the "transformations" directory and its contents to the directory where the aeon module is installed. Entries for Rager have been added to the __init__.py file, so it must be replaced. The Rager class inherits from the BaseCollectionTransformer class of aeon.

Script with an example of use ("example validation.py") is also provided.

# Polish Sign Language (PSL_extended) dataset
PSL_extended is a sample dataset created as a benchmark for Rager and other time series classification methods.

PSL_extended, consists of nine Polish Sign Language gestures: "good morning", "goodbye", "greetings", "please", "thank you", "why?", "yes", "no", and "I don't understand". Gestures were performed by six people seated in a fixed position relative to an RGB camera. Each person performed each gesture five times. Half of the dataset (people with numbers 1, 3 and 4) are used as the training subset by default, while the other half (people with numbers 2, 4 and 6) are used as the testing subset. Then, the Pose Detection module of the MediaPipe library was used to generate 3D coordinates of body landmarks (characteristic body parts). Only the torso and arm landmarks were used, while the head and legs were skipped due to the specificity of sign language.

PSL_extended is available in text format ("PSL_extended dataset" directory) and in Python NumPy format ("PSL_extended.npz").

# Requirements
Python modules needed to run the provided example script using Rager are:

* numpy
* aeon
* sklearn
* time (optional - for time measurements)

The modules can be installed using the pip command.
