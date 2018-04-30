% % DIAGNOSIS OF SMALL CELL LUNG CANCER THROUGH ARTIFICIAL NEURAL NETWORKS
%  In the code below, we perform the following tasks:
%  ------------
%   
%  1- Load X-ray image data of diganosed small cell lung cancer patients.
%  2- A sample image (obtained from cancer image archive has 1951*2000  
%     samples. So, sampling is performed to take a limited number of
%     samples only and make the program reasonable.
%  3- Neural network structure is defined.
%  4- Initially, random weights are selected and output is calculated    
%     through feedforward mechanism and cost function is evaluated.
%  5- Then, through back-propagation mechanism, we canclulate error in 
%     neurons of all hidden layers. This is a self-correcting phenomenon.
%  6- Regularization is performed to avoid overfitting.
%  7- We test the model on 20% images to find out the accuracy of model.

%Setting up the environment for project
clear ; close all; clc

%Defining paramters for neural network.
input_layer_size  = 400;  % 20x20 Input Images of Small Cell Lung X-rays 
hidden_layer_size = 25;   % 25 hidden units
num_labels = 3;          % 1 = benign or non-malignant disease;
                         % 2 = malignant, primary lung cancer
                         % 3 = malignant metastatic
                          
                         
%  The dataset contains 12 images of patients with small cell lung cancer.

% % Load Training Data
fprintf('Loading Data ...\n')
load('data1.mat');
X=images;

fprintf('Program paused. Press enter to continue.\n');
pause;

% Load the pre-initialized neural network weights.

fprintf('\nLoading Saved Neural Network Parameters ...\n')
 
% % Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');
% 
% % Unroll parameters 

Theta2=Theta2(1:3,:)
nn_params = [Theta1(:) ; Theta2(:)];

%Implementing feedforward propagation and calculating cost

fprintf('\nFeedforward Using Neural Network ...\n')

% Initially set regularization paramter to zero.
lambda = 0;

%% TODO
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded above): %f '...
         '\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Now, regularizing the above forward-feedback propagation to avoid
%  overfitting

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Setting regularization paramter to 1
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

%fprintf(['Cost at parameters with regularization: %f '], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  Calculating the sigmoid gradient

fprintf('\nEvaluating sigmoid gradient...\n')


g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%  Now we will implment a twolayer neural network that classifies X-rays in
%  3 categories. 

fprintf('\nInitializing Neural Network Parameters ...\n')

%Randomly initalizing the weights
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  Now, implementing backpropogation to analyze with neurons contribute
%  heavily to error.

fprintf('\nChecking Backpropagation... \n');


%  Check gradients by running checkNNGradients which is used to validate 
% gradient descent
checkNNGradients;
 
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Now, once again, we implement regularization to avoid overfitting.

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% fprintf('\nTraining Neural Network... \n')

% Optimset iterarates gradient descent 50 times
options = optimset('MaxIter', 50);

lambda = 1;

% Creating "short hand" for the cost function to be minimized

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%  First we predict on our training set.
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

load('data3.mat')

pred = predict(Theta1, Theta2, images2);
y1=[2 ;3];
fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y1)) * 100);
