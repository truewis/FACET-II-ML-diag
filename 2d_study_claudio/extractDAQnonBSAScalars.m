%path =  '/nas/nas-li20-pm00/E327/2021/20210813/E327_00326';
%load([path,'/E327_00326.mat'])
function [nonbsaScalarData,nonbsaVarPVs] = extractDAQnonBSAScalars(data_struct)

dataScalars = data_struct.scalars;
idx = dataScalars.common_index;
% Find the names of all BSA lists
fNames = fieldnames(dataScalars);
nonbsaListNames = regexp(fNames,'nonBSA_List_');
for ij=1:length(nonbsaListNames);nonbsaLists(ij) = isequal(nonbsaListNames{ij},1);end
isnonBSA = find(nonbsaLists);% Index of BSA lists in data_struct.scalars
nonbsaScalarData = [];
nonbsaVarPVs = [];

% Loop through all BSA lists and concatenate data
for nList = 1:length(isnonBSA)
    listNum = isnonBSA(nList);
    nonbsaList = dataScalars.(fNames{listNum}); % Go through each BSA list    
    varNames = fieldnames(nonbsaList);% Extract the BSA variable names
    
    for nVar = 1:length(varNames)
        varData = nonbsaList.(varNames{nVar});
        varData = varData(idx);
        varData(isnan(varData))=0;
        nonbsaListData(nVar,:) = varData;        
    end
    
    nonbsaVarPVs = cat(1,nonbsaVarPVs,varNames);% Concatenate PV names in one cell array
    nonbsaScalarData = cat(1,nonbsaScalarData,nonbsaListData);% Concatenate Output in one big BSA array

    clearvars nonbsaListData varNames
end