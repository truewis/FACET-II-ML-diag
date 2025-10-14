%path =  '/nas/nas-li20-pm00/E327/2021/20210813/E327_00326';
%load([path,'/E327_00326.mat'])
function [bsaScalarData,bsaVarPVs] = extractDAQBSAScalars(data_struct)

dataScalars = data_struct.scalars;
idx = dataScalars.common_index;
% Find the names of all BSA lists
fNames = fieldnames(dataScalars);
bsaListNames = regexp(fNames,'BSA_List_');
for ij=1:length(bsaListNames);bsaLists(ij) = isequal(bsaListNames{ij},1);end
isBSA = find(bsaLists);% Index of BSA lists in data_struct.scalars
bsaScalarData = [];
bsaVarPVs = [];

% Loop through all BSA lists and concatenate data
for nList = 1:length(isBSA)
    listNum = isBSA(nList);
    bsaList = dataScalars.(fNames{listNum}); % Go through each BSA list    
    varNames = fieldnames(bsaList);% Extract the BSA variable names
    emptyVarsIdx = [];

    for nVar = 1:length(varNames)
        varData = bsaList.(varNames{nVar});
        if isempty(varData)% Ignore variables with empty data
            emptyVarsIdx = [emptyVarsIdx nVar];
            continue;
        else
        varData = varData(idx);
        varData(isnan(varData))=0;
        bsaListData(nVar,:) = varData;    
        end
    end
    
    bsaVarPVs = cat(1,bsaVarPVs,varNames(~ismember(1:numel(varNames), emptyVarsIdx)));% Concatenate PV names in one cell array, ignoring empty variables
    bsaScalarData = cat(1,bsaScalarData,bsaListData);% Concatenate Output in one big BSA array

    clearvars bsaListData varNames
end