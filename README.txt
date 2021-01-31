Unsupervised clustering of mnist data (images of hand written digits) and subsequent training of a convolutional neural network.
Only 10 labeled data samples are used (one for each label type 0-9) to map each cluster to a label.

Tested in Windows 10 and Ubuntu 18.04.2 LTS.
Takes around 15-21 minutes to run in CPU-mode on an Intel Core i7 6700 2.6GHz, 8GB RAM with Windows 10 and Ubuntu.

####################################
###########Use example (Ubuntu)#####
#Get the code
git clone https://github.com/bosse89/unsupervised_oneshot_mnist
#Create a virual environment
virtualenv -p `which python` unsupervised_oneshot_mnist/venvProj1
#Start your virual environment
source unsupervised_oneshot_mnist/venvProj1/bin/activate
#Go in to project folder
cd unsupervised_oneshot_mnist
#Install required packages
pip3 install -r requirements.txt
#Run the main script
python main.py
####################################

Method:
Clustering: PCA -> K-nearest neighbors -> Louvain clustering
One shot learning: Clusters are mapped to labels using only 1 labeled data sample per label type (0-9).
Classification: convolutional neural network

Results:
Clusters have V-measure (homogeneity and completeness) = 0.9-0.95
Convolutional neural network accuracy on test data: 90-95%

Acknolwdgements:
Code from the following websites were used to inspire and accelerate this code project:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
https://github.com/nathandelara/MNIST-unsupervised
