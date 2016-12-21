function plotsomweightdist(net,inputs)
% plotsomweightdist(net,inputs) plots a SOM where each neuron is
% represented by a pie chart showing the relative sizes of the weight
% vector for that neuron. The size (area) of the pie chart is proportional
% to the number of hits for the neuron.

% Get the dimensions
d = net.layers{1}.dimensions;
n = d(1);
m = d(2);

% sqrt(3)/2 is a useful constant (the height of an equilateral triangle
% with base length 1)
sq32 = sqrt(3)/2;
% Determine the scaling factor to fit the m-by-n grid of pie charts into
% the figure window
scl = max(1+(m-1)*sq32,n+0.5);

% Get the hits for the neurons, and scale (for use later when resizing the
% pie charts). Because positions are given [x0 y0 dx dy], resizing also
% requires shifting. If dx and dy are scaled by a factor of s, then x0 and
% y0 need to be shifted by (1-s)/2.
pred = net(inputs);
hits = sum(pred,2);
hits = sqrt(hits/max(hits));
shift = (1-hits)/2;

% Get the weights for the neurons.
W = net.IW{1};
W(W==0) = realmin;  % just in case (PIE doesn't like 0s)

% An offset to center the plots in the figure window
% (One of these will be 0, depending on the aspect ratio of the SOM)
xoff = 0.5*(scl - (n + 0.5*rem(m+1,2)));
yoff = 0.5*(scl - (1 + (m-1)*sq32));

% Make an array of axes
ax = gobjects(n,m);
for k = 1:m
    for j = 1:n
        % Make a pie chart in the (j,k)th axes
        % ((j,k) ->  idx in linear index)
        idx = n*(k-1)+j;
        ax(j,k) = axes;
        p = pie(W(idx,:));
        % Get rid of the text annotations
        % (pie chart = n patches + n text annotations, interleaved)
        delete(p(2:2:end))
        % Standardize the limits and maximize area used by actual graph
        ax(j,k).XLim = [-1 1];
        ax(j,k).YLim = [-1 1];
        % Shift and scale the axes to put the pie chart in the position of
        % the (j,k)th neuron
        pos = [xoff + (j-1) + 0.5*rem(k+1,2),yoff + (k-1)*sq32,1,1]/scl;
        % Scale by hits (and shift to keep centered)
        pos(1:2) = pos(1:2) + shift(idx)*pos(3:4);
        pos(3:4) = hits(idx)*pos(3:4);
        % Update the axis position
        ax(j,k).Position = pos;
    end
end
