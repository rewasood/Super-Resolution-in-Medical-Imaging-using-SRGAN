import scipy
import numpy as np
import skimage

import scipy.misc
import skimage.measure

image_list = ['27', '78', '403', '414', '480', '579', '587', '664', '711', '715', '756', '771', '788', '793', '826', '947', '994', '1076', '1097', '1099', '1141', '1197', '1263', '1320', '1389', '1463', '1563']
#image_list = ['27', '78', '403', '414', '480', '579']
gnd_truth_hr_image_path = 'Data/MRI/PaperTestData/HR_gnd/'
generated_hr_image_path = 'Data/MRI/PaperTestData/HR_gen/'

avg_psnr = 0
avg_ssim = 0

for im in image_list:
	gnd_truth_hr_img = scipy.misc.imread(gnd_truth_hr_image_path+'valid_hr-id-'+im+'.png', mode='L')
	generated_hr_img = scipy.misc.imread(generated_hr_image_path+'valid_hr_gen-id-'+im+'.png', mode='L')

	# print out PSNR and SSIM
	psnr_i = skimage.measure.compare_psnr(gnd_truth_hr_img, generated_hr_img)
	ssim_i = skimage.measure.compare_ssim(gnd_truth_hr_img, generated_hr_img, data_range=generated_hr_img.max() - generated_hr_img.min())
	print('PSNR = ' + str(psnr_i) + ', SSIM = ' + str(ssim_i))

	avg_psnr += psnr_i
	avg_ssim += ssim_i

avg_psnr /= len(image_list)
avg_ssim /= len(image_list)

print('Average PSNR = ' + str(avg_psnr))
print('Average SSIM = ' + str(avg_ssim))


# resize ground truth to (384x384) image
#gnd_truth_hr_img = scipy.misc.imread(gnd_truth_hr_image_path, mode='L')
#gnd_truth_hr_img_resized = scipy.misc.imresize(gnd_truth_hr_img, [384, 384], interp='bicubic', mode='L')

# read generated (384x384) image
#generated_hr_img = scipy.misc.imread(generated_hr_image_path, mode='L')

# print out PSNR
#print(skimage.measure.compare_psnr(gnd_truth_hr_img_resized, generated_hr_img))

# print out SSIM
#print(skimage.measure.compare_ssim(gnd_truth_hr_img_resized, generated_hr_img, data_range=generated_hr_img.max() - generated_hr_img.min()))
