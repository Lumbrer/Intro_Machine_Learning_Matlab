function [TableOut] = cleanTable(TableIn,varargin)
%cleanTable - Clear invalid data from a Table array
% Removal of NaN, zero values or negative values.
%
%This function will return an empty table array when input is not of class
%table
%
% Syntax:  [TableOut] = cleanTable(TableIn,Options)
%
% Inputs:
%    Table - Table array to be cleaned
%    Options - Cell array of string specifying the criteria to clear rows
%    from the input table. Additional inputs will be ignored.
%       'NaN' - Remove NaN instances
%       'Inf' - Remove Inf values
%       'Zero' - Remove zero values
%       'Negative' - Remove negative values
%       'Outlier' - Remove elements out of mean +- 3*std for numeric
%       columns. Notice the data is assumed to fit a normal distribution
%       'Undefined' - Remove undefined values from categorical columns
%       'Nat' - Remove not a time values from datetime class variables
%       'All' - Apply all above options.
% - DEFAULT IF NO OPTIONS ARE PROVIDED IS Options={'NaN','Inf','Undefined'}
%
% Outputs:
%    TableOut - Table after elimination of invalid data
%
%
% Example:
%    T_clean=cleanTable(DataTable);
%    T_clean=cleanTable(DataTable,{'All'});
%    T_clean=cleanTable(DataTable,{'NaN','Zero'});
%
%
% See also: ~

% $Author: Lumbrer $    $Date: 2016/11/02 $    $Revision: 0.1 $
% Copyright: Francisco Lumbreras

default={'NaN','Inf','Undefined'};


if istable(TableIn)
    
    if nargin>=2&&iscell(varargin{1})
        % Extract options
        options=varargin{1};
    else
        % Default to base case
        options=default;
    end
    
    %Extract Table dimensions
    [nr,nc]=size(TableIn);
    
    %Split Numeric and Non-Numeric
    Tnum=table;
    Tdes=table; %Categorical and other
    Ttime=table; %Datetime
    
    
    %Split data in case categorical data is present in the input table
    for j=1:nc
        if(isnumeric(TableIn{:,j}))
            Tnum=[Tnum TableIn(:,j)];
        elseif (isdatetime(TableIn{:,j}))
            Ttime=[Ttime TableIn(:,j)];
        else
            Tdes=[Tdes TableIn(:,j)];
        end
    end
    
    
    %Check for existance of NaN and if required eliminate such entries
    if any(strcmp('NaN',options))||any(strcmp('All',options))
        idnan=any(isnan(Tnum{:,:}),2);
        if ~isempty(idnan)
            Tnum(idnan,:)=[];
            if ~isempty(Tdes)
                Tdes(idnan,:)=[];
            end
            if ~isempty(Ttime)
                Ttime(idnan,:)=[];
            end
        end
        clear idnan
        
    end
    
    
    %Check for existance of Inf numeric values and if required eliminate
    %such entries
    if any(strcmp('Inf',options))||any(strcmp('All',options))
        idinf=any(isinf(Tnum{:,:}),2);
        if ~isempty(idinf)
            Tnum(idinf,:)=[];
            if ~isempty(Tdes)
                Tdes(idinf,:)=[];
            end
            if ~isempty(Ttime)
                Ttime(idinf,:)=[];
            end
        end
        clear idinf
        
    end
    
    
    %Check for existance of zeros in the numeric values and if required
    %eliminate such entries
    if any(strcmp('Zero',options))||any(strcmp('All',options))
        Values=Tnum{:,:};
        idzeros=Values==0;
        idtoremove=false(size(idzeros,1),1);
        for i=1:size(Values,1)
            if any(idzeros(i,:))
                idtoremove(i)=true;
            end
        end
        if any(idtoremove)
            Tnum(idtoremove,:)=[];
            if ~isempty(Tdes)
                Tdes(idtoremove,:)=[];
            end
            if ~isempty(Ttime)
                Ttime(idtoremove,:)=[];
            end
        end
        clear idtoremove idzeros Values
    end
    
    
    %Check for existance of negative numbers in the numeric values and if
    %required eliminate such entries
    if any(strcmp('Negative',options))||any(strcmp('All',options))
        Values=Tnum{:,:};
        idneg=Values<0;
        idtoremove=false(size(idneg,1),1);
        for i=1:size(Values,1)
            if any(idneg(i,:))
                idtoremove(i)=true;
            end
        end
        if any(idtoremove)
            Tnum(idtoremove,:)=[];
            if ~isempty(Tdes)
                Tdes(idtoremove,:)=[];
            end
            if ~isempty(Ttime)
                Ttime(idtoremove,:)=[];
            end
        end
        clear idtoremove idneg Values
    end
    
    
    %Check for existance of outliers in the numeric values and if
    %required eliminate such entries
    if any(strcmp('Outlier',options))||any(strcmp('All',options))
        Values=Tnum{:,:};%Take numerical values
        [r,c]=size(Values);
        %Calculate mean and std for each variable (columns)
        % A point is an outlier if outside of interval mean+-3*std
        idtoremove=false(r,1);
        for j=1:c
            mu=mean(Values(:,j),'omitnan');
            sd=std(Values(:,j),'omitnan');
            columnvalues=Values(:,j);
            outliers=columnvalues>mu+3*sd|columnvalues<mu-3*sd;
            idtoremove=outliers|idtoremove;
        end
        if any(idtoremove)
            Tnum(idtoremove,:)=[];
            if ~isempty(Tdes)
                Tdes(idtoremove,:)=[];
            end
            if ~isempty(Ttime)
                Ttime(idtoremove,:)=[];
            end
        end
        clear idtoremove mu sd Values columnvalues
    end
    
    
    %Check for existance of undefined categorical values in the descriptive
    % table and remove these
    if any(strcmp('Undefined',options))||any(strcmp('All',options))
        if ~isempty(Tdes)
            idtoremove=false(size(Tdes,1),1);
            variables=Tdes.Properties.VariableNames;
            for ij=1:length(variables)
                if(iscategorical(Tdes.(variables{ij})))
                    idtoremove=isundefined(Tdes.(variables{ij}))|idtoremove;
                end
            end
            if any(idtoremove)
                Tdes(idtoremove,:)=[];
                if ~isempty(Tnum)
                    Tnum(idtoremove,:)=[];
                end
                if ~isempty(Ttime)
                    Ttime(idtoremove,:)=[];
                end
                
            end
            clear idtoremove variables
        end
    end
    
    %Check for existance of Nat and if required eliminate such entries
    if any(strcmp('Nat',options))||any(strcmp('All',options))
        if ~isempty(Ttime)
            idnat=any(isnat(Ttime{:,:}),2);
            if ~isempty(idnat)
                Ttime(idnat,:)=[];
                if ~isempty(Tnum)
                    Tnum(idnat,:)=[];
                end
                if ~isempty(Tdes)
                    Tdes(idnat,:)=[];
                end
            end
            clear idnat
        end
        
    end
    
    %Produce the outut table once cleaned from "bad data"
    
    TableOut=[Ttime Tnum Tdes];
else
    TableOut=table;
end

end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

