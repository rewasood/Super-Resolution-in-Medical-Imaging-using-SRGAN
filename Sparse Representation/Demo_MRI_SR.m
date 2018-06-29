% =========================================================================
% Simple demo codes for image super-resolution via sparse representation
%
% Reference
%   J. Yang et al. Image super-resolution as sparse representation of raw
%   image patches. CVPR 2008.
%   J. Yang et al. Image super-resolution via sparse representation. IEEE 
%   Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010
%
% Jianchao Yang
% ECE Department, University of Illinois at Urbana-Champaign
% For any questions, send email to jyang29@uiuc.edu
% =========================================================================

clear all; clc;

%image_list = {'The_Big_Bang_Theory1_S19E01_0248_wanted.png';
%    'The_Simpsons_S19E01_0003_wanted.png'
%    };

image_list = {'27', '78', '403', '414', '480', '579', '587', '664', '711', '715', '756', '771', '788', '793', '826', '947', '994', '1076', '1097', '1099', '1141', '1197', '1263', '1320', '1389', '1463', '1563'};
%image_list = {'ProstateDx-01-0001__09-23-2008-MRI PROSTATE WITH AND WITHOUT CONTRAST-00237__401-T2WTSECOR-06613__000000.png'};
for i = 1:length(image_list)
    disp(sprintf('Working on image # %d of %d ...', i, length(image_list)));
    fn_full = fullfile(sprintf('Data/MRI/PaperTestData/HR_gen/valid_hr_gen-id-%s.png',image_list{i}));
    if exist(fn_full,'file')
        continue;
    end
    % read test image
    im_l = imread(sprintf('Data/MRI/PaperTestData/LR/valid_lr-id-%s.png',image_list{i}));
    %im_l = imread(sprintf('Data/MRI/LowResMRI_Images/%s',image_list{i}));
    
    % hacky code to copy image into 3 RGB channels by kchoutag:
    if(ndims(im_l) == 2)
        im_new = zeros(size(im_l, 1), size(im_l, 2), 3);
        im_new(:,:,1) = im_l;
        im_new(:,:,2) = im_l;
        im_new(:,:,3) = im_l;
        im_l = uint8(im_new);
    end

    % set parameters
    lambda = 0.2;                   % sparsity regularization
    overlap = 4;                    % the more overlap the better (patch size 5x5)
    up_scale = 2;                   % scaling factor, depending on the trained dictionary
    maxIter = 20;                   % if 0, do not use backprojection

    % load dictionary
    load('Dictionary/MRI/D_100_0.15_5.mat');

    % change color space, work on illuminance only
    im_l_ycbcr = rgb2ycbcr(im_l);
    im_l_y = im_l_ycbcr(:, :, 1);
    im_l_cb = im_l_ycbcr(:, :, 2);
    im_l_cr = im_l_ycbcr(:, :, 3);

    % image super-resolution based on sparse representation
    [im_h_y] = ScSR(im_l_y, 2, Dh, Dl, lambda, overlap);
    [im_h_y] = ScSR(im_h_y, 2, Dh, Dl, lambda, overlap);
    [im_h_y] = backprojection(im_h_y, im_l_y, maxIter);

    % upscale the chrominance simply by "bicubic" 
    [nrow, ncol] = size(im_h_y);

    im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
    im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');

    im_h_ycbcr = zeros([nrow, ncol, 3]);
    im_h_ycbcr(:, :, 1) = im_h_y;
    im_h_ycbcr(:, :, 2) = im_h_cb;
    im_h_ycbcr(:, :, 3) = im_h_cr;
    im_h = ycbcr2rgb(uint8(im_h_ycbcr));
    % bicubic interpolation for reference
    im_b = imresize(im_l, [nrow, ncol], 'bicubic');

    %save image
    
    fid = fopen(fn_full,'w+');
    fclose(fid);
    imwrite(im_h,fn_full);
end %while
