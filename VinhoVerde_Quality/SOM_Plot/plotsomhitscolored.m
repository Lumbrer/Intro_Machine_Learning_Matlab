function plotsomhitscolored(net,inputs,ctgry,cmap,lgnd)
% plotsomhitscolored(net,inputs,cat) plots a SOM with each neuron showing
% the number of hits, and colored by the categories of the hits. The color
% of each neuron is the average of the colors of the categories, weighted
% by the number of hits on that neuron from each category.
%
% The colors of the categories are chosen from the default (parula)
% colormap. plotsomhitscolored(net,inputs,cat,cmap) uses colors defined by
% the colormap specified in the string CMAP.
%
% Use plotsomhitscolored(net,inputs,cat,cmap,'nolegend') to make the plot
% with no legend. Use plotsomhitscolored(net,inputs,cat,[],'nolegend') to
% specify default colors.
%
% Example:
% load iris_dataset
% net = selforgmap([5 5]);
% net = train(net,irisInputs);
% plotsomhits(net,irisInputs)
% species = categorical([1 2 3]*irisTargets,1:3,{'setosa','versicolor','virginica'});
% figure
% plotsomhitscolored(net,irisInputs,species);


% Get number of categories
if iscategorical(ctgry)
    catnames = categories(ctgry);
else
    catnames = unique(ctgry);
end
m = numel(catnames);
% Get colors
if (nargin < 4) || isempty(cmap)
    % Set default colors
    colors = parula(m);
else
    % Check how colors were specified
    if isnumeric(cmap)
        % Numeric matrix => must be cm-by-3
        % Ideally, the number of colors (cm) should be the same as the
        % number of categories (m)
        [cm,cn] = size(cmap);
        if (cn ~= 3)
            error('Numeric colors must be given as an n-by-3 matrix')
        elseif (cm < m)
            error(['Not enough colors -- ',num2str(cm),...
                ' colors given for ',num2str(m),' categories.'])
        elseif (cm > m)
            warning(['More colors provided (',num2str(cm),...
                ') than needed for ',num2str(m),...
                ' categories.  Extra colors will be ignored.'])
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

if (nargin < 5)
    lgnd = true;
elseif isequal(lgnd,'nolegend')  || isequal(lgnd,false)
    lgnd = false;
else
    error('Unknown legend option')
end

% Get index of predicted neuron for each input
pred = vec2ind(net(inputs));
% Count the number of inputs of each category assigned to each neuron
% Get the categories back b/c some neurons may have 0 hits
[hits,~,~,cats] = crosstab(pred,ctgry);

% Number of neurons in the SOM
n = net.output.size;

% Need to do some fiddling to deal with (the possibility of) neurons with
% no hits.  The matrix HITS should be n-by-m, but may have fewer rows.
% So, make an n-by-m matrix and index into the rows according to the neuron
% values returned by crosstab (in CATS).
neuronnum = str2double(cats(:,1));
allhits = zeros(n,m);
allhits(neuronnum,:) = hits;
% Normalized by row sums to give percentages (note: neurons with no hits
% will result in rows of NaNs)
hitperc = bsxfun(@rdivide,allhits,sum(allhits,2));

% Make the standard SOM hits plot
plotsomhits(net,inputs)
% Get the patches representing the neuron hits
ax = gca;
% The graph consists of 3n objects: two patches and one text for each
% neuron. To display correctly, they have to be ordered such that the
% patches representing the hits are in the middle.
hitpatch = get(ax,'Children');
hitpatch = hitpatch((n+1):(2*n));
% For each patch, the new color is the weighted average of the m category
% colors. Use matrix multiplication to get these in one go
% Also max out at 1, in case randoff results in a color value of 1+eps
newpatchcolors = min(hitperc*colors,1);
% Replace NaNs with 1s => white patches (doesn't really matter because they
% should have no size anyway)
newpatchcolors(isnan(newpatchcolors)) = 1;

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
    legend(legpatches,catnames,'Location','EastOutside')
end
