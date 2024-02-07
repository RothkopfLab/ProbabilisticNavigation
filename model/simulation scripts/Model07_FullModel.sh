#!/bin/bash
# -------------------------------
#SBATCH -J 07-MC_FullModel
#SBATCH --mail-type=END
#SBATCH -A project02264
#SBATCH -e /work/scratch/iv55otop/MC-07-Full-Model.err
#SBATCH -o /work/scratch/iv55otop/MC-07-Full-Model.out
#SBATCH -n 96
#SBATCH -C i01
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3000
#SBATCH -t 03:00:00
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

cd /home/iv55otop/00-Projects/probabilistic-navigation
source env/bin/activate

python simulate_trials.py -experiment $experiment -n_landmarks $n_landmarks -n_samples $n_samples -enable_target_jitter $target_jitter -output_folder $output_folder -individual_participant_file $individual_participant_file -motion_noise_enabled True -observation_noise_enabled True -representation_noise_enabled True -individual_participant_file_type "$individual_participant_file_type"  -parameter_set "$random" -model_type "Full Model"
python evaluate_experiment.py -input /work/scratch/iv55otop/ray_tune_runs/$output_folder -output ray_tune_runs/$output_folder


