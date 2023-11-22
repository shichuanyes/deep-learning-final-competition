## short script to download data from google drive
module load rclone/1.60.1 
rclone copy wbg231_drive:fall_2023/competition/Dataset_Student_V2.zip ./
unzip Dataset_Student_V2.zip 
rm Dataset_Student_V2.zip 