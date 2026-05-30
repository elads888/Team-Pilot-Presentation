## Environment
Install needed packages with:
conda env create -f environment.yaml
Then activate with:
conda activate team_pilot

## Data
Download the OCMR cardiac CINE dataset from [this url](https://www.ocmr.info).
Then perform augmentations as performed in Multi-PILOT:
Run python augment.py --data-path <path to your downloaded OCMR dataset>


## Experiment Reproducability
To train the base model for a given <n> number of shots (RF excitations) with learned trajectories run:
python train_vst.py --augment --trajectory-learning --n-shots <n>


To train the base model for a given <n> number of shots (RF excitations) without learned trajectories run:
python train_vst.py --augment --n-shots <n>

You can choose a specific trajectory initialization (say, for recreating the non-learned trajectory baselines for TEAM-PILOT) by passing the --initialization argument.
Values can be set to radial or golden to recreate baselines.



Use --resume and --checkpoint to load an existing model

To run trajectory refinement with regularizatin weight <lambda> over a trained model located in path <path> with <n> shots run:
python finetune_trajs.py --augment --bound-weight <la,bda> --resume --checkpoint <path> --n-shots <n>

In our experiments lambda was set to 5.
