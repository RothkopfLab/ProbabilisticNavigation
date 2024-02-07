#!/bin/bash
mkdir /work/scratch/iv55otop/ray_tune_runs/$2 
mkdir ../ray_tune_runs/$2 

#Nardini 2008 
sbatch Model00_Zero_Model.sh -n 350 -o $2/nardini2008_adults_mc_00_zero_model_params_$1 -e nardini2008 -l 3 -j False -p True -r 33 
sbatch Model01_Motor_Variability.sh -n 350 -o $2/nardini2008_adults_mc_01_motor_model_params_$1 -e nardini2008 -l 3 -j False -p True -r 40
sbatch Model02-Perceptual_Variability.sh -n 350 -o $2/nardini2008_adults_mc_02_perceptual_model_params_$1 -e nardini2008 -l 3 -j False -p True -r 22
sbatch Model03_Representation_Variability.sh -n 350 -o $2/nardini2008_adults_mc_03_representation_model_params_$1 -e nardini2008 -l 3 -j False -p True -r 32
sbatch Model04_Motor_Perceptual_Variability.sh -n 350 -o $2/nardini2008_adults_mc_04_motorperceptual_model_params_$1 -e nardini2008 -l 3 -j False -p True -r 6
sbatch Model05_Motor_Representation_Variability.sh -n 350 -o $2/nardini2008_adults_mc_05_motorrepresentation_model_params_$1 -e nardini2008 -l 3 -j False -p True -r 16
sbatch Model06_Representation_Perceptual_Variability.sh -n 350 -o $2/nardini2008_adults_mc_06_representationperceptual_model_params_$1 -e nardini2008 -l 3 -j False -p True -r 32
sbatch Model07_FullModel.sh -n 350 -o $2/nardini2008_adults_mc_07_full_model_params_$1 -e nardini2008 -l 3 -j False -p True -r 16

#Chen 2017 
#3 Chen2017 3 Landmarks
for individual_file in _1 _2 _3 _4; do
	echo $individual_file
	sbatch Model00_Zero_Model.sh -n 350 -o $2/chen2017_rich_mc_00_zero_model_params_$1 -e chen2017 -l 3 -j False -p True -i $individual_file -r 33 
   	sbatch Model01_Motor_Variability.sh -n 350 -o $2/chen2017_rich_mc_01_motor_model_params_$1 -e chen2017 -l 3 -j False -p True -i $individual_file -r 40	
	sbatch Model02-Perceptual_Variability.sh -n 350 -o $2/chen2017_rich_mc_02_perceptual_model_params_$1 -e chen2017 -l 3 -j False -p True -i $individual_file -r 22
	sbatch Model03_Representation_Variability.sh -n 350 -o $2/chen2017_rich_mc_03_representation_model_params_$1 -e chen2017 -l 3 -j False -p True -i $individual_file -r 32
	sbatch Model04_Motor_Perceptual_Variability.sh -n 350 -o $2/chen2017_rich_mc_04_motorperceptual_model_params_$1 -e chen2017 -l 3 -j False -p True -i $individual_file -r 6
	sbatch Model05_Motor_Representation_Variability.sh -n 350 -o $2/chen2017_rich_mc_05_motorrepresentation_model_params_$1 -e chen2017 -l 3 -j False -p True -i $individual_file -r 16
	sbatch Model06_Representation_Perceptual_Variability.sh -n 350 -o $2/chen2017_rich_mc_06_representationperceptual_model_params_$1 -e chen2017 -l 3 -j False -p True -i $individual_file -r 32
	sbatch Model07_FullModel.sh -n 350 -o $2/chen2017_rich_mc_07_full_model_params_$1 -e chen2017 -l 3 -j False -p True -i $individual_file -r 16
done 


for individual_file in _1 _2 _3; do 
	echo $individual_file
	sbatch Model00_Zero_Model.sh -n 350 -o $2/chen2017_poor_mc_00_zero_model_params_$1 -e chen2017 -l 1 -j False -p True -i $individual_file -r 33
	sbatch Model01_Motor_Variability.sh -n 350 -o $2/chen2017_poor_mc_01_motor_model_params_$1 -e chen2017 -l 1 -j False -p True -i $individual_file -r 40
	sbatch Model02-Perceptual_Variability.sh -n 350 -o $2/chen2017_poor_mc_02_perceptual_model_params_$1 -e chen2017 -l 1 -j False -p True -i $individual_file -r 22 
	sbatch Model03_Representation_Variability.sh -n 350 -o $2/chen2017_poor_mc_03_representation_model_params_$1 -e chen2017 -l 1 -j False -p True -i $individual_file -r 32
	sbatch Model04_Motor_Perceptual_Variability.sh -n 350 -o $2/chen2017_poor_mc_04_motorperceptual_model_params_$1 -e chen2017 -l 1 -j False -p True -i $individual_file -r 6 
	sbatch Model05_Motor_Representation_Variability.sh -n 350 -o $2/chen2017_poor_mc_05_motorrepresentation_model_params_$1 -e chen2017 -l 1 -j False -p True -i $individual_file -r 16
	sbatch Model06_Representation_Perceptual_Variability.sh -n 350 -o $2/chen2017_poor_mc_06_representationperceptual_model_params_$1 -e chen2017 -l 1 -j False -p True -i $individual_file -r 32
	sbatch Model07_FullModel.sh -n 350 -o $2/chen2017_poor_mc_07_full_model_params_$1 -e chen2017 -l 1 -j False -p True -i $individual_file -r 16 
done 



#Zhao 2015 
#proximal 
for individual_file in _1 _2;  do 
	#regular 
	#echo $individual_file
	sbatch Model00_Zero_Model.sh -n 350 -o $2/zhao2015a_proximal_mc_00_zero_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_base$individual_file -r 33
	sbatch Model01_Motor_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_01_motor_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_base$individual_file -r 40 
	sbatch Model02-Perceptual_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_02_perceptual_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_base$individual_file -r 22 
	sbatch Model03_Representation_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_03_representation_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_base$individual_file -r 32
	sbatch Model04_Motor_Perceptual_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_04_motorperceptual_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_base$individual_file -r 6 
	sbatch Model05_Motor_Representation_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_05_motorrepresentation_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_base$individual_file -r 16
	sbatch Model06_Representation_Perceptual_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_06_representationperceptual_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_base$individual_file -r 32 
	sbatch Model07_FullModel.sh -n 350 -o $2/zhao2015a_proximal_mc_07_full_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_base$individual_file -r 16 

	#conflict 
	sbatch Model00_Zero_Model.sh -n 350 -o $2/zhao2015a_proximal_mc_00_zero_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_conflict$individual_file -r 33
	sbatch Model01_Motor_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_01_motor_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_conflict$individual_file -r 40
	sbatch Model02-Perceptual_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_02_perceptual_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_conflict$individual_file -r 22
	sbatch Model03_Representation_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_03_representation_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_conflict$individual_file -r 32
	sbatch Model04_Motor_Perceptual_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_04_motorperceptual_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_conflict$individual_file -r 6 
	sbatch Model05_Motor_Representation_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_05_motorrepresentation_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_conflict$individual_file  -r 16
	sbatch Model06_Representation_Perceptual_Variability.sh -n 350 -o $2/zhao2015a_proximal_mc_06_representationperceptual_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_conflict$individual_file -r 32 
	sbatch Model07_FullModel.sh -n 350 -o $2/zhao2015a_proximal_mc_07_full_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _proximal_conflict$individual_file -r 16 
done 

# distal 
#
for individual_file in _1 _2 _3 _4; do 
	#regular 
#	echo $individual_file
	sbatch Model00_Zero_Model.sh -n 350 -o $2/zhao2015a_distal_mc_00_zero_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_base$individual_file -r 33 
	sbatch Model01_Motor_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_01_motor_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_base$individual_file -r 40
	sbatch Model02-Perceptual_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_02_perceptual_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_base$individual_file -r 22 
	sbatch Model03_Representation_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_03_representation_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_base$individual_file -r 32
	sbatch Model04_Motor_Perceptual_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_04_motorperceptual_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_base$individual_file -r 6 
	sbatch Model05_Motor_Representation_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_05_motorrepresentation_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_base$individual_file  -r 16
	sbatch Model06_Representation_Perceptual_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_06_representationperceptual_model_params_$1 -e zhao2015a -l 3 -j False -p True -i _distal_base$individual_file  -r 32
	sbatch Model07_FullModel.sh -n 350 -o $2/zhao2015a_distal_mc_07_full_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_base$individual_file -r 16

	#conflict 
	sbatch Model00_Zero_Model.sh -n 350 -o $2/zhao2015a_distal_mc_00_zero_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_conflict$individual_file -r 33 
	sbatch Model01_Motor_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_01_motor_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_conflict$individual_file -r 40
	sbatch Model02-Perceptual_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_02_perceptual_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_conflict$individual_file -r 22
	sbatch Model03_Representation_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_03_representation_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_conflict$individual_file -r 32
	sbatch Model04_Motor_Perceptual_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_04_motorperceptual_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_conflict$individual_file -r 6 
	sbatch Model05_Motor_Representation_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_05_motorrepresentation_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_conflict$individual_file -r 16
	sbatch Model06_Representation_Perceptual_Variability.sh -n 350 -o $2/zhao2015a_distal_mc_06_representationperceptual_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_conflict$individual_file -r 32
	sbatch Model07_FullModel.sh -n 350 -o $2/zhao2015a_distal_mc_07_full_model_params_$1 -e zhao2015a -l 3 -j True -p True -i _distal_conflict$individual_file -r 16 
done
