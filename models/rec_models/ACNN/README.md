# ACNN-K-SPACE

Paper: Adaptive convolutional neural networks for k-space data interpolation in fast magnetic resonance imaging [https://arxiv.org/pdf/2006.01385.pdf].

Train and test:


Run "pip install -r requirement.txt" to install customized library.


Run "python -m visdom.server -port 8098" to start a visdom server (for visualization).


Download the Stanford MRI data from [here](https://drive.google.com/drive/folders/1T-ZXNNyBrKfXpO23kjYLf2T9lohDGy_I?usp=sharing) and edit dataset path in config.txt.For training your own data, please organize your data in the stanford dataset form (each file in subfolder is a k-space slice)


Edit other customization parameters in config.txt.


Run main.py to train and test the model.
