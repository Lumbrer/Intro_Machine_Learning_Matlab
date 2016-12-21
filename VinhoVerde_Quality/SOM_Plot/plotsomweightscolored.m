function plotsomweightscolored(net,inputs,cmap,lgnd)
% plotsomweightscolored(net,inputs) plots a SOM with each neuron showing
% the number of hits, and colored by the weights. The color of each neuron
% is the average of the colors of the input variables, weighted by the
% weight vector of that neuron.
%
% The colors of the weights are chosen from the default (parula) colormap.
% plotsomweightscolored(net,inputs,cmap) uses colors defined by the
% colormap specified in the string CMAP.
%
% Use plotsomweightscolored(net,inputs,cmap,'nolegend') to make the plot
% with no legend. Use plotsomweightscolored(net,inputs,[],'nolegend') to
% specify default colors.
%
% Example:
% load iris_dataset
% net = selforgmap([5 5]);
% net = train(net,irisInputs);
% plotsomhits(net,irisInputs)
% figure
% plotsomweightscolored(net,irisInputs);

% Extract the weights
W = net.IW{1};
% Normalize weights for each neuron
W = bsxfun(@rdivide,W,sum(W,2));

% Number of neurons in the SOM
[n,m] = size(W);

% Get colors
if (nargin < 3) || isempty(cmap)
    % Set default colors
    colors = parula(m);
else
    % Check how colors were specified
    if isnumeric(cmap)
        % Numeric matrix => must be cm-by-3
        % Ideally, the number of colors (cm) should be the same as the
        % number of variables (m)
        [cm,cn] = size(cmap);
        if (cn ~= 3)
            error('Numeric colors must be given as an n-by-3 matrix')
        elseif (cm < m)
            error(['Not enough colors -- ',num2str(cm),...
                ' colors given for ',num2str(m),' input variables.'])
        elseif (cm > m)
            warning(['More colors provided (',num2str(cm),...
                ') than needed for ',num2str(m),...
                ' input variables.  Extra colors will be ignored.'])
        end
        % Success
        colors = cmap(1:m,:);
    elseif ischar(cmap)
        % String => use the function with that name to create the colors
        try
            f = str2func(cmap);
            colors = f(m);
        catch
            error(['Colormap "',cmap,'" not recognized'])
        end
    else
        % Neither numeric nor string => fail
        error('Colormap must be a string or an n-by-3 matrix')
    end
end

if (nargin < 4)
    lgnd = true;
elseif isequal(lgnd,'nolegend')  || isequal(lgnd,false)
    lgnd = false;
else
    error('Unknown legend option')
end


% Make the standard SOM hits plot
plotsomhits(net,inputs)
% Get the patches representing the neuron hits
ax = gca;
% The graph consists of 3n objects: two patches and one text for each
% neuron. To display correctly, they have to be ordered such that the
% patches representing the hits are in the middle.
hitpatch = get(ax,'Children');
hitpatch = hitpatch((n+1):(2*n));
% For each patch, the new color is the weighted average of the colors of
% the m neuron weights. Use matrix multiplication to get these in one go
% Also max out at 1, in case randoff results in a color value of 1+eps
newpatchcolors = min(W*colors,1);

% Change the patch colors. Note that patches are in reverse order.
for k = 1:n
    set(hitpatch(k),'FaceColor',newpatchcolors(n+1-k,:))
end

% Add legend (unless specified)
if lgnd
    % Total and utter hack: add m patches, one of each color, hidden away
    for k = 1:m
        legpatches(k) = patch([0 0 0.1 0.1],[0 0.1 0.1 0],colors(k,:),...
            'Parent',ax); %#ok<AGROW>
    end
    % Reorder these patchs to the background where they won't be seen
    axch = get(ax,'Children');
    set(ax,'Children',axch([(1:3*n)+m,1:m]))
    % Now use those patches in the legend :)
    legend(legpatches,cellstr(num2str((1:m)')),'Location','EastOutside')
end
