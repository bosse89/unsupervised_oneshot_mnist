Unsupervised clustering of mnist data and subsequent training of a convolutional neural network.
Only 10 labeled data samples are used (one for each label type 0-9) to map each cluster to a label.

Tested in Windows 10 and Ubuntu 18.04.2 LTS.

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

Acknolwdgements:
Code from the following websites were used to inspire and accelerate this code project:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
https://github.com/nathandelara/MNIST-unsupervised
