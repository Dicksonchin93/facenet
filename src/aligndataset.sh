for N in {1..7}; do python align/align_dataset_mtcnn.py /media/dcek/storage/MS_celeb_1/Cropped/Images/ /media/dcek/storage/MS_celeb_1/Manual_align_crop/ --image_size 182 --margin 44 --random_order --gpu_memory_fraction 0.13 & done


#python align/align_dataset_mtcnn.py /media/dcek/storage/MS_celeb_1/Cropped/Images/ /media/dcek/storage/MS_celeb_1/Manual_align_crop/ --image_size 182 --margin 44 --random_order
