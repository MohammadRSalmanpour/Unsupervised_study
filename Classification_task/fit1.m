function obj = fit(training, group, varargin)
% Not intended to be called directly. Use FITCNB to fit a NaiveBayes.
%
%   See also FITNAIVEBAYES.

%   Copyright 2008-2015 The MathWorks, Inc.

warning(message('stats:obsolete:ReplaceThisWith','NaiveBayes','fitcnb'));

narginchk(2,Inf);

if ~isnumeric(training) 
    error(message('stats:NaiveBayes:fit:TrainingBadType'));
end

if ~isreal(training)
    error(message('stats:NaiveBayes:fit:TrainingComplexType'));
end

obj = NaiveBayes;

[gindex,obj.ClassNames, obj.ClassLevels] = grp2idx(group);
n = size(training,1);
if n == 0 ||size(gindex,1) == 0
    error(message('stats:NaiveBayes:fit:EmptyData'));
end

if n ~= size(gindex,1);
    error(message('stats:NaiveBayes:fit:MismatchedSize'));
end
 
nans = isnan(gindex);
if any(nans)
    training(nans,:) = [];
    gindex(nans) = [];
end

obj.NClasses = length(obj.ClassNames);
obj.ClassSize = hist(gindex,1: obj.NClasses);
obj.CIsNonEmpty = (obj.ClassSize > 0)';
obj.NonEmptyClasses =find(obj.ClassSize>0);

obj.LUsedClasses = length(obj.NonEmptyClasses);
if obj.NClasses > obj.LUsedClasses
    warning(message('stats:NaiveBayes:fit:EmptyGroups'));
end

[n, obj.NDims]= size(training);
if n == 0
    error(message('stats:NaiveBayes:fit:NoData'));
end

% Parse input and error check
pnames = {'distribution' 'prior'   'kswidth'    'kssupport' 'kstype'};
dflts =  {'normal'       'empirical' []         []           []};
[obj.Dist,prior, kernelWidth,obj.KernelSupport, obj.KernelType] ...
    = internal.stats.parseArgs(pnames, dflts, varargin{:});

if isempty(prior) || (ischar(prior) && strncmpi(prior,'empirical',length(prior)))
    obj.Prior = obj.ClassSize(:)' / sum(obj.ClassSize);
elseif ischar(prior) && strncmpi(prior,'uniform',length(prior))
    obj.Prior = ones(1, obj.NClasses) / obj.NClasses;
    % Explicit prior
elseif isnumeric(prior)
    if ~isvector(prior) || length(prior) ~= obj.NClasses
        error(message('stats:NaiveBayes:fit:ScalarPriorBadSize'));
    end
    obj.Prior = prior;
elseif isstruct(prior)
    if ~isfield(prior,'group') || ~isfield(prior,'prob')
        error(message('stats:NaiveBayes:fit:PriorBadFields'));
    end
    [pgindex,pgroups] = grp2idx(prior.group);
    
    ord =NaN(1,obj.NClasses);
    for i = 1:obj.NClasses
        j = find(strcmp(obj.ClassNames(i), pgroups(pgindex)));
        if isempty(j)
            error(message('stats:NaiveBayes:fit:PriorBadGroup'));
        elseif numel(j) > 1
             error(message('stats:NaiveBayes:fit:PriorDupClasses'));
        else
            ord(i) = j;
        end
    end
    obj.Prior = prior.prob(ord);
    
else
    error(message('stats:NaiveBayes:fit:BadPriorType'));
end

obj.Prior(obj.ClassSize==0) = 0;
if any(obj.Prior < 0) || sum(obj.Prior) == 0
    error(message('stats:NaiveBayes:fit:BadPrior'));
end
obj.Prior = obj.Prior(:)' / sum(obj.Prior); % normalize the row vector

if ischar(obj.Dist)
    obj.Dist = cellstr(obj.Dist); %convert a single string to a cell
elseif ~iscell(obj.Dist)
    error(message('stats:NaiveBayes:fit:BadDist'));
end
% distribution list must be a vector
if ~isvector(obj.Dist)
      error(message('stats:NaiveBayes:fit:BadDist'));
end
if numel(obj.Dist) ~= 1% ~isscalar(obj.Dist)
    if length(obj.Dist) ~= obj.NDims
        error(message('stats:NaiveBayes:fit:BadDistVec'));
    end
    
    try
        u = unique(obj.Dist);
    catch ME
        if isequal(ME.identifier,'MATLAB:UNIQUE:InputClass')
            error(message('stats:NaiveBayes:fit:BadDist'));
        else
            rethrow(ME);
        end
    end
    %if all the distribution are same, make it a scalar cell
    if numel(u) == 1 &&  ~strncmpi(u,'mn',length(u{1}))
        obj.Dist = u;
    end
else
     if ~ischar(obj.Dist{1})
        error(message('stats:NaiveBayes:fit:BadDist'));
     end
end

distNames = {'normal','mvmn','kernel','mn'};
if numel(obj.Dist) == 1 %isscalar(obj.Dist)
    %i = strmatch(lower(obj.Dist),distNames);
    i = find(strncmpi(obj.Dist{1},distNames,length(obj.Dist{1})));
    if isempty(i)
        error(message('stats:NaiveBayes:fit:UnknownScalarDist',obj.Dist{1}));
    elseif numel(i) > 1
        error(message('stats:NaiveBayes:fit:AmbiguousScalarDist',obj.Dist{1}));
    elseif i == 1
        obj.GaussianFS = true(1,obj.NDims);
    elseif i ==2 %'mvmn'
        obj.MVMNFS = true(1,obj.NDims);
    elseif i == 3 %'kernel'
        obj.KernelFS = true(1,obj.NDims);
    end %
    obj.Dist = distNames(i);
    
else %obj.Dist is a vector
    obj.GaussianFS = false(1,obj.NDims); % flag for Gaussian features
    obj.MVMNFS = false(1,obj.NDims); % flag for multivariate multinomial features
    obj.KernelFS = false(1,obj.NDims);   % flag for kernel features

    for d = 1:obj.NDims
        curDist =obj.Dist{d};
        
        i = find(strncmpi(curDist,distNames,length(curDist)));
        if isempty(i)
            error(message('stats:NaiveBayes:fit:UnknownDistVector',curDist,d));
        elseif numel(i)>1
            error(message('stats:NaiveBayes:fit:AmbiguousDistVector',curDist,d));
        elseif i==4
            error(message('stats:NaiveBayes:fit:BadDistMN'));
        elseif i==1
            obj.GaussianFS(d) = true;
        elseif i==2
            obj.MVMNFS(d) = true;
        elseif i==3
            obj.KernelFS(d) = true;
        end
        obj.Dist{d} = distNames{i};
    end %loop over d
    
 u = unique(obj.Dist);
    if length(u) == 1
        obj.Dist = u;
    end
end

if isscalar(obj.Dist) && strcmp(obj.Dist,'mn')
    nans = any(isnan(training),2);%remove rows with any NaN
    %remove rows with invalid values
    trBad =  any(training< 0 |  training ~= round(training), 2);
    if any(trBad)
        warning(message('stats:NaiveBayes:fit:BadDataforMN'));
    end
    t = nans | trBad;
    if any(t)
        training(t,:) = [];
        gindex(t) = [];
    end
else
    nans = all(isnan(training),2);%remove rows with all NaNs
    if any(nans)
        training(nans,:) = [];
        gindex(nans) = [];
    end
    
    for k = obj.NonEmptyClasses
        groupI = (gindex == k);
        if sum(groupI) == 0
            error(message('stats:NaiveBayes:fit:NoDataInEachClass'));
        end
        nanCols =  all(isnan(training(groupI,:)),1);
        if any(nanCols)
            nanCols = strtrim(sprintf('%d ',find(nanCols)));
            error(message('stats:NaiveBayes:fit:TrainingAllNaN', obj.ClassNames{ k }, nanCols));
        end
    end
end

%process the kernel options
if any(obj.KernelFS)
    if ~isempty(kernelWidth)
        if isnumeric(kernelWidth)
            %check the size of kernel width
            [wd1, wd2]=size(kernelWidth);
            if(wd1 ~= 1 && wd1 ~= obj.NClasses) || (wd2 ~= 1 && wd2 ~= obj.NDims)
                error(message('stats:NaiveBayes:fit:ScalarKernelWidthSizeBad'));
            end
            obj.KernelWidth = kernelWidth;
            
        elseif isstruct(kernelWidth)
            if ~isfield(kernelWidth,'group') || ~isfield(kernelWidth,'width')
                error(message('stats:NaiveBayes:fit:KernelWidthBadFields'));
            end
            
            if ~isnumeric(kernelWidth.width)
                error(message('stats:NaiveBayes:fit:KernelWidthNonNumeric'));
            end
            
            [kwgindex,kwgroups] = grp2idx(kernelWidth.group);
            if size(kernelWidth.width,1) ~= length(kwgroups);
                error(message('stats:NaiveBayes:fit:KernelWidthRowSizeBad'));
            end
            if size(kernelWidth.width,2) ~= 1 &&...
                    size(kernelWidth.width,2) ~= obj.NDims;
                error(message('stats:NaiveBayes:fit:KernelWidthColumnSizeBad'));
            end
            ord = NaN(1,obj.NClasses);
            
            for i = 1:obj.NClasses
                j = find(strcmp(obj.ClassNames(i), kwgroups(kwgindex)));
                if isempty(j)
                    error(message('stats:NaiveBayes:fit:KernelWidthBadGroup'));
                elseif numel(j) > 1
                    error(message('stats:NaiveBayes:fit:KernelWidthDupClasses'));
                else
                    ord(i) = j; 
                end
            end
            obj.KernelWidth = kernelWidth.width(ord,:);
        else
            error(message('stats:NaiveBayes:fit:InvalidKernelWidth'));
        end
        
        %check the validity of kernel width.
        if size(obj.KernelWidth,2) > 1
            kwtemp = obj.KernelWidth(:,obj.KernelFS);
        else
            kwtemp = obj.KernelWidth;
        end
        
        if size(obj.KernelWidth,1) > 1
            kwtemp = kwtemp(obj.NonEmptyClasses,:);
        end
        
        kwtemp = kwtemp(:);
        
        if  any(~isfinite(kwtemp)) || any(kwtemp <= 0)
                error(message('stats:NaiveBayes:BadKSWidth'));
        end
        
        
    end % ~isempty(kernelWidth)
    
    if ~isempty(obj.KernelSupport)
        
        if iscell(obj.KernelSupport) 
            if isscalar(obj.KernelSupport) %allow a cell with only one element
                obj.KernelSupport = validSupport(obj.KernelSupport{1});
            else
                if ~isvector(obj.KernelSupport) || length(obj.KernelSupport) ~= obj.NDims
                    error(message('stats:NaiveBayes:fit:BadSupport'));
                end
                %check each kernelsupport
                supporttemp = obj.KernelSupport(obj.KernelFS);
                for i = 1: numel(supporttemp)
                    supporttemp{i}= validSupport(supporttemp{i});
                end
                obj.KernelSupport(obj.KernelFS) = supporttemp;
            end
        else
            obj.KernelSupport = validSupport(obj.KernelSupport);
        end
    else
        obj.KernelSupport = 'unbounded';
    end % ~isempty(obj.KernelSupport)
    
    if ~isempty(obj.KernelType)
        if ischar(obj.KernelType)
            obj.KernelType = cellstr(obj.KernelType);
        elseif ~iscell(obj.KernelType)
            error(message('stats:NaiveBayes:fit:BadKSType'));
        end
        if ~isvector(obj.KernelType)
            error(message('stats:NaiveBayes:fit:BadKSType'));
        end
        
        if isscalar(obj.KernelType)
            obj.KernelType = validKernelType(obj.KernelType{1});
        else
            %check the length of vector kernelType
            if length(obj.KernelType) ~= obj.NDims
                error(message('stats:NaiveBayes:fit:KSTypeBadSize'));
            end
            
            kernelTypeTemp = obj.KernelType(obj.KernelFS);
            for i = 1: numel(kernelTypeTemp)
                kernelTypeTemp{i}= validKernelType(kernelTypeTemp{i});
            end
            obj.KernelType(obj.KernelFS) = kernelTypeTemp;
        end
    else
        obj.KernelType = 'normal';
    end
    
end

obj.Params = cell(obj.NClasses, obj.NDims);

%Start Fit
if isscalar(obj.Dist)
    switch obj.Dist{:}
        case 'mn'
            obj =  mnfit(obj,training, gindex);
        case 'normal'
            obj = gaussianFit(obj, training, gindex);
        case 'mvmn'
            obj = mvmnFit(obj, training,gindex);
        case 'kernel'
            obj = kernelFit(obj,training, gindex);
    end
else
    if any(obj.GaussianFS)
        obj = gaussianFit(obj, training, gindex);
    end
    if any(obj.MVMNFS)
        obj = mvmnFit(obj, training,gindex);
    end
    if any(obj.KernelFS)
        obj = kernelFit(obj,training, gindex);
    end
    
end

end %fit

%--------------------------------------
%estimate parameters using Gaussian distribution
function obj = gaussianFit(obj, training, gidx)
for i = obj.NonEmptyClasses
    groupI = (gidx == i);
    
    gsize = sum(~isnan(training(groupI,obj.GaussianFS)),1);
    if any(gsize < 2)
        error(message('stats:NaiveBayes:fit:NoEnoughDataForGaussian'));
    end
    mu = nanmean(training(groupI,obj.GaussianFS));
    sigma = nanstd(training(groupI,obj.GaussianFS));
    badCols = sigma <= gsize * eps(max(sigma));
    if any(badCols)
        badCols = sprintf('%d ',find(badCols));
        error(message('stats:NaiveBayes:fit:BadVariance', badCols, obj.ClassNames{ i }));
    end
    obj.Params(i,obj.GaussianFS) = mat2cell([mu;sigma],2,...
        ones(1,sum(obj.GaussianFS)));
    %Each cell is a 2-by-1 vector, the first element is the mean,
    %and the second element is the standard deviation.
end
end %function gaussianFit

%-------------------------------------------
%Use kernel density estimate
function obj = kernelFit(obj, training,gidx)

kdfsidx = find(obj.KernelFS);
kw2=[];
if ~isempty(obj.KernelWidth)
    [kwrLen,kwcLen] = size(obj.KernelWidth);
    kw = obj.KernelWidth;
    if kwrLen == 1
        kw = repmat(kw, [obj.NClasses,1]);
    end
    if kwcLen == 1
        kw = repmat(kw, [1,obj.NDims]);
    end
end

for i = obj.NonEmptyClasses
    groupI = (gidx == i);
    
    for j = kdfsidx
        if iscell(obj.KernelSupport)
            kssupport = obj.KernelSupport{j};
        else
            kssupport = obj.KernelSupport;
        end
        if iscell(obj.KernelType)
            kstype = obj.KernelType{j};
        else
            kstype = obj.KernelType;
        end
        
        data = training(groupI,j);
        nans = isnan(data);
        if any(nans)
            data(nans)=[];
            if size(data,1) == 0
                error(message('stats:NaiveBayes:fit:NoEnoughDataForKernel'));
            end
        end
        
        if ~isempty(obj.KernelWidth)
            kw2 = kw(i, j);
        end
        
        obj.Params{i,j} = ...
            fitdist(data,'kernel', 'width',kw2,...
            'support',kssupport,'kernel',kstype);
        
    end
    
    
end
end


%---------------------------
%estimate the parameters using multivariate multinomial
function obj = mvmnFit(obj, training, gidx)

mvmnfsIdx = find(obj.MVMNFS);
d = sum(obj.MVMNFS);
mvmnParams = cell(obj.NClasses,d);
obj.UniqVal = cell(1,d);
for j = 1: d
    data = training(:,mvmnfsIdx(j));
    gidx2 = gidx;
    nans = isnan(data);
    if any(nans)
        data(nans)=[];
        gidx2(nans)=[];
        
    end
    obj.UniqVal{j} = unique(data);
    for i = obj.NonEmptyClasses
        groupI = (gidx2 == i);
        if sum(groupI) == 0
            error(message('stats:NaiveBayes:fit:NoEnoughDataForMVMN'));
        end
        p = histc(data(groupI),obj.UniqVal{j});
        %Add one count for each discrete value of the training data to avoid zero probability
        p= (p+1)/(size(data(groupI),1) +length(obj.UniqVal{1,j}));
        mvmnParams(i,j) = {p(:)};
    end
end

obj.Params(:,obj.MVMNFS) = mvmnParams;

end

%-----------------------------------------------------
% perform Multinomial fit
function obj =  mnfit(obj,training, gidx)
d = size(training,2);
for k = obj.NonEmptyClasses
    groupI = (gidx == k);
    if sum(groupI) == 0
        error(message('stats:NaiveBayes:fit:NoDataInEachClass'));
    end
    
    pw = sum(training(groupI,:),1);
    pw = (pw+1)/(sum(pw)+d);
    %Add one count for  each feature to avoid zero probability
    obj.Params(k,:)= num2cell(pw);% mat2cell(pw,1,ones(1,d));
end
end

%-----------------------------------
%check the validity of kernelsupport
function  kssupport = validSupport(kssupport)

badSupport = false;
if ischar(kssupport) && size(kssupport,1)==1
    supportName = {'unbounded' 'positive'};
  
    i = find(strncmpi(kssupport,supportName,length(kssupport)));
    if isempty(i)
        badSupport = true;
    else
        kssupport = supportName{i};
    end
    
elseif ~(isnumeric(kssupport) && numel(kssupport)==2 ...
        && all(isfinite(kssupport)) && kssupport(1) < kssupport(2))
    badSupport = true;
end
if badSupport
    error(message('stats:NaiveBayes:fit:BadSupport'));
end
end

%----------------------------------------------------------------
%check the validity of kernel Type
function type = validKernelType(type)
typeNames ={'normal' , 'box', 'triangle', 'epanechnikov'};

if ~ischar(type)
    error(message('stats:NaiveBayes:fit:BadKSType'));
end

i = find(strncmpi(type,typeNames, length(type)));
if isempty(i)
    error(message('stats:NaiveBayes:fit:UnknownKSType',type));   
end
type= typeNames{i};
end
