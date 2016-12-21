function [TableOut] = normTable(TableIn,varargin)
%normTable - normalise all numeric values in a table (Rescaling to a
%different range or standardisation according to chosen options)
%
%
%This function will return an empty table array when input is not of class
%table
%
% Syntax:  [TableOut] = normTable(TableIn,Options)
%
% Inputs:
%    Table - Table array to be normalised
%    Options - Cell array of string specifying the criteria for normalised 
%    rows from the input table. Additional inputs will be ignored.
%       'Range' - Reduce Min - Max range to [0,1] (Rescaling)
%       'Centre' - Shift centre to zero mean and unitary std 
%    (standardisation)
% - DEFAULT IF NO OPTIONS ARE PROVIDED IS Options={'Centre'}
%
% Outputs:
%    TableOut - Table after normalisation of numeric values
%
%
% Example:
%    T_norm=normTable(DataTable);
%    T_norm=normTable(DataTable,{'Range'});
%
%
% See also: cleanTable.m (used to remove NaN form data)

% $Author: Lumbrer $    $Date: 2016/11/02 $    $Revision: 0.1 $
% Copyright: Francisco Lumbreras

default={'Centre'};

if istable(TableIn)
    
     if nargin>=2&&iscell(varargin{1})
        % Extract options
        options=varargin{1};
    else
        % Default to base case
        options=default;
    end
    
    TableToProcess=cleanTable(TableIn);
    
    %Extract Table dimensions
    [nr,nc]=size(TableToProcess);
    
    %Split Numeric and Non-Numeric
    Tnum=table;
    Tdes=table; %Categorical and other
    
    
    %Split data in case categorical data is present in the input table
    for j=1:nc
        if(isnumeric(TableToProcess{:,j}))
            Tnum=[Tnum TableToProcess(:,j)];
        else
            Tdes=[Tdes TableToProcess(:,j)];
        end
    end
    
    Values=Tnum{:,:};
    
    if any(strcmp('Centre',options)) 

    Values=zscore(Values);
    
    else
        [nrn,ncn]=size(Values);
        for v=1:ncn
            minval=min(Values(:,v));
            maxval=max(Values(:,v));
            range=maxval-minval;
            Values(:,v)=(Values(:,v)-minval)./range;
        end
    end
    Tnum{:,:}=Values;
    
    TableOut=[Tnum Tdes];
    
else
    TableOut=table;
end

end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------