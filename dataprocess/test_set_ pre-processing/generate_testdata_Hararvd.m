clear 
clc
close all

dataset = 'Hararvd';
% upscale = 3;
upscale = 4;

% savePath = ['D:/test/',dataset,'/',num2str(upscale)]; %save test set  to "savePath"
% savePath = ['E:/fusion_project/dataset/Harvard/Hararvd/test',num2str(upscale)]; %save test set  to "savePath"
savePath = ['H:/fusion_project/dataset/Harvard/Hararvd_x4/test/']; %save test set  to "savePath"


if ~exist(savePath, 'dir')
    mkdir(savePath)
end

%% obtian all the original hyperspectral image
% srPath = 'E:\data\HyperSR\Hararvd\test';
% srPath = 'E:\fusion_project\dataset\Harvard\mat';
srPath = 'H:\fusion_project\dataset\Harvard\Hararvdori\testori';
srFile=fullfile(srPath);
% srdirOutput=dir(fullfile(srFile));
srdirOutput=dir(fullfile(srFile,'*.mat'));
srfileNames={srdirOutput.name}';
number = length(srfileNames)


for index = 1 : number
    name = char(srfileNames(index));
    if(isequal(name,'.')||... % remove the two hidden folders that come with the system
           isequal(name,'..'))
               continue;
    end
    disp(['-----deal with:',num2str(index),'----name:',name]); 
    load([srPath,'/',name])
    data =ref;
    clear lbl
    clear ref

    %% normalization
    data = data/(1.0*max(max(max(data))));
    data = data(1:512, 1:512,:);
    
    %% obtian HR and LR hyperspectral image    
    img = reshape(data, size(data,1)*size(data,2), 31);
    HR = modcrop(data, upscale);
    LR = imresize(HR,1/upscale,'bicubic'); %LR  

    save([savePath,'/',name], 'HR', 'LR')

    clear HR
    clear LR

end
