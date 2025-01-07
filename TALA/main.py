from trainers.train import Trainer

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

if __name__ == "__main__":

    # ========  Experiments Phase ================
    parser.add_argument('--phase',               default='train',         type=str, help='train, test')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name',               default='TALA_2',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='TALA',               type=str, help='MCD, NO_ADAPT, Deep_Coral, MMDA, DANN, CDAN, DIRT, DSAN, HoMM, CoDATS, AdvSKM, SASA, CoTMix, TARGET_ONLY')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default=r'../data',                  type=str, help='Path containing datase2t')
    parser.add_argument('--dataset',                default='HHAR_SA',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default= "cuda",                   type=str, help='cpu or cuda')

    # arguments
    args = parser.parse_args()

    # create trainier object
    trainer = Trainer(args)

    # train and test
    if args.phase == 'train':
        trainer.fit()
    elif args.phase == 'test':
        trainer.test()



#TODO:
# 1- Change the naming of the functions ---> ( Done)
# 2- Change the algorithms following DCORAL --> (Done)
# 3- Keep one trainer for both train and test -->(Done)
# 4- Create the new joint loader that consider the all possible batches --> Done
# 5- Implement Lower/Upper Bound Approach --> Done
# 6- Add the best hparams --> Done
# 7- Add pretrain based methods (ADDA, MCD, MDD)
