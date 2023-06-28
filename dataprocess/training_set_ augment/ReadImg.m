function [imgUint,imgDouble] = ReadImg(imgFolderPath,imgSize,imgType)
%UNTITLED2 读取同一个文件夹下的同一类型的同样大小的图片，以多维数组的形式返回
%   inpout arguement one
%       imgFolderPath:图片所在文件夹绝对路径，以反斜杠结尾，
%        demo:imgFoldpath = 'G:/ZhaoFY/data/cave/chart_and_stuffed_toy_ms/';
%   input arguement two
%       imgSize:图片的宽和高
%   inpout arguement three
%       imgType:图片类型，jpg,png
%   output argument 
%       imgUint:uint格式的多维数组，未标准化
%       imgDouble:double格式的多维数组，为标准化

imgFoldpath = imgFolderPath;
imgDir=dir([imgFoldpath '*.' imgType]);
img = cell(1,length(imgDir));

imgUint = zeros([imgSize length(imgDir)],'uint16');%创建uint8的多维数组
imgDouble = zeros(size(imgUint));%创建double的多维数组

for i = 1:length(imgDir)
    imgUint(:,:,i) = imread([imgFoldpath imgDir(i).name]); %读取每张图片
    imgDouble(:,:,i) = double(imgUint(:,:,i));
end