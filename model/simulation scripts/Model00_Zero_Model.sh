#!/bin/bash
# -------------------------------
#SBATCH -J 00-MC-ZeroModel
#SBATCH --mail-type=END
#SBATCH -A project02264
#SBATCH -e /work/scratch/iv55otop/MC-00-Zero-Model.err
#SBATCH -o /work/scratch/iv55otop/MC-00-Zero-Model.out
#SBATCH -n 96
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3000
#SBATCH -t 04:00:00
# -------------------------------
while getopts :n:o:f:e:l:j:p:i:r: flag
do
    case "${flag}" in
        n) n_samples=${OPTARG};;
        o) output_folder=${OPTARG};;
        f) fullname=${OPTARG};;
        e) experiment=${OPTARG};;
        l) n_landmarks=${OPTARG};;
        j) target_jitter=${OPTARG};;
        p) individual_participant_file=${OPTARG};;
        i) individual_participant_file_type=${OPTARG};;
        r) random=${OPTARG}
    esac
done

echo $n_samples
echo $output_folder

echo $PWD
cd /home/iv55otop/00-Projects/probabilistic-navigation
source env/bin/activate
#python simulate_trials.py -experiment $experiment -n_landmarks $n_landmarks -n_samples $n_samples -enable_target_jitter $target_jitter -output_folder $output_folder -individual_participant_file $individual_participant_file -motion_noise_enabled False -motion_cov 0.00001,0.00001,0.00001,0.00001 -observation_noise_enabled False -observation_cov 0.00001,0.00001,0.00001,0.00001  -representation_noise_enabled False -representation_cov 0.00001 -representation_random_walk False -representation_noise_type None -individual_participant_file_type $individual_participant_file_type


python simulate_trials.py -experiment $experiment -n_landmarks $n_landmarks -n_samples $n_samples -enable_target_jitter $target_jitter -output_folder $output_folder -individual_participant_file $individual_participant_file -motion_noise_enabled False -observation_noise_enabled False -representation_noise_enabled False -representation_noise_type None -individual_participant_file_type "$individual_participant_file_type" -parameter_set "$random" -model_type "Zero Model"
python evaluate_experiment.py -input /work/scratch/iv55otop/ray_tune_runs/$output_folder -output ray_tune_runs/$output_folder
