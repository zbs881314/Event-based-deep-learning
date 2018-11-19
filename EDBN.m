clear all
%% Load paths
addpath(genpath('.'));

%% Load data
load mnist_uint8;

% Convert data and rescale between 0 and 0.2
train_x = double(train_x) / 255 * 0.2;
test_x  = double(test_x)  / 255 * 0.2;
train_y = double(train_y) * 0.2;
test_y  = double(test_y)  * 0.2;

%% Train network
% Setup
rand('seed', 42);
clear edbn opts;
edbn.sizes = [784 100 100 10];
opts.numepochs = 6;

[edbn, opts] = edbnsetup(edbn, opts);

% Train
fprintf('Beginning training.\n');



edbn = edbntrain(edbn, train_x, opts);
% Use supervised training on the top layer
edbn = edbntoptrain(edbn, train_x, opts, train_y);
a1=edbn.erbm{1}.W;
b1=a1>=0;
c1=a1.*b1;
Max1=max(max(c1))
[x1 y1]=find(c1==Max1)
d1=uint8(round(255*a1/Max1));
a2=edbn.erbm{2}.W;
b2=a2>=0;
c2=a2.*b2;
Max2=max(max(c2))
[x2 y2]=find(c2==Max2)
d2=uint8(round(255*a2/Max2));
a3=edbn.erbm{3}.W;
b3=a3>=0;
c3=a3.*b3;
Max3=max(max(c3))
[x3 y3]=find(c3==Max3)
d3=uint8(round(255*a3/Max3));

% Show results
figure;
visualize(edbn.erbm{1}.W');   %  Visualize the RBM weights
er = edbntest (edbn, train_x, train_y);
fprintf('Scored: %2.2f\n', (1-er)*100);

%% Show the EDBN in action
spike_list = live_edbn(edbn, test_x(1, :), opts);
output_idxs = (spike_list.layers == numel(edbn.sizes));
sum(sum(output_idxs>0))

figure(2); clf;
hist(spike_list.addrs(output_idxs) - 1, 0:edbn.sizes(end));
title('Label Layer Classification Spikes');
%% Export to xml
