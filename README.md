# Siamese-networks
- I have created this repository to experiment with differrent CNN architectures for similarity matching between two images using Siamese networks.
- **Siamese.py** will create a siamese network with two sister VGG-16s sharing weights. The session graph will then be stored as a protobuf file named as **siam.pb**
- **viz_graph.py** will create tensorboard event logs and write it into a directory **LOGS** which we can use to visualize the graph with tensorboard. 
