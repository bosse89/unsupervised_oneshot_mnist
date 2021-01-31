Unsupervised clustering of mnist data and subsequent training of a convolutional neural network.
Only 10 labeled data samples are used (one for each label type 0-9) to map each cluster to a label.

Tested in Windows 10 and Ubuntu 18.04.2 LTS.

####################################
###########Use example (Ubuntu)#####
#Create a virual environment
virtualenv -p /usr/bin/python project1/venvProj1
#Start your virual environment
source project1/venvProj1/bin/activate
#Get the code in to your project folder.
git clone https://github.com/bosse89/unsupervised_oneshot_mnist project1
#Go in to project folder
cd project1
#Install required packages
pip3 install -r requirements.txt
#Run the main script
python main.py
####################################

Acknolwdgements:
Code from the following websites were used to inspire and accelerate this code project:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
https://github.com/nathandelara/MNIST-unsupervised