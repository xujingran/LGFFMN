clc
clc
clear 
close all
%%
% imgFoldpath = 'G:/ZhaoFY/data/cave/chart_and_stuffed_toy_ms/';
% imagename  = ["balloons_ms"];
image = 'watercolors_ms';
imgFoldpath = ['E:/fusion_project/dataset/CAVE/cave1/' image '/'];
imgDir=dir([imgFoldpath '*.png']);
img = cell(1,length(imgDir));
%%
% for i = 1:length(imgDir)          % 遍历结构体就可以一一处理图片了
%     img{i} = imread([imgFoldpath imgDir(i).name]); %读取每张图片
% end
% 
% for i = 1:length(imgDir)          % 遍历结构体就可以一一处理图片了
%     imshow(img{i}); %读取每张图片
% end
%%
[imgUint,imgDouble] = ReadImg(imgFoldpath,[512,512],'png');
for i = 1:length(imgDir)          % 遍历结构体就可以一一处理图片了
    imshow(imgUint(:,:,i)); %读取每张图片
end
%%
% unitname = imagename + '_uint16'
unit = [image '_uint16'];
eval([unit, '= imgUint;']);

% unit = imgUint;
% cave_toy_ms_double = imgDouble;
% doublename = imagename;
double = [image];
eval([double, '= imgDouble;']);
% save cave_toy_ms.mat cave_toy_ms_uint16 cave_toy_ms_double 
path = 'E:/fusion_project/dataset/CAVE/mat/';
save ([path, double '.mat'],  'watercolors_ms') 