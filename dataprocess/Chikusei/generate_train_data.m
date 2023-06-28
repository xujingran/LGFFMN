% % path = '/Data/Chikusei/Train/';
% % path = 'H:\fusion_project\dataset\dataprocess\matlab_code\Chikusei\train\';
% path = 'H:\fusion_project\dataset\hsisr_dataset\Chikusei_x4\trainori\';
% patch_size = 128;
% % stride = 64;
% stride = 32;
% % factor = 0.125;
% factor = 0.25;
% % save_dir = "/Data/Chikusei/train_dataset_x8/";
% % save_dir = "H:\fusion_project\dataset\dataprocess\matlab_code\Chikusei\train\train_dataset_x8\";
% save_dir = 'H:\fusion_project\dataset\hsisr_dataset\Chikusei_x4\train32\';



path = 'H:\fusion_project\dataset\hsisr_dataset\Chikusei_x8\trainori\';
patch_size = 256;
stride = 64;
% stride = 32;
factor = 0.125;
% factor = 0.25;
% save_dir = "/Data/Chikusei/train_dataset_x8/";
% save_dir = "H:\fusion_project\dataset\dataprocess\matlab_code\Chikusei\train\train_dataset_x8\";
save_dir = 'H:\fusion_project\dataset\hsisr_dataset\Chikusei_x8\train32\';


file_folder=fullfile(path);
file_list=dir(fullfile(file_folder,'*.mat'));
file_names={file_list.name};


% store cropped images in folders
for i = 1:1:numel(file_names)
    name = file_names{i};
    name = name(1:end-4);
    load(strcat(path,file_names{i}));
    crop_image(img, patch_size, stride, factor, name, save_dir);
end