import numpy as np
import matplotlib.pyplot as plt

# eta, beta1, beta2

eta_range = np.logspace(0.0001, 0.1, 500)
beta1_range = np.logspace(0.1, 0.999, 500)
beta2_range = np.logspace(0.1, 0.9999, 500)

etas = np.random.choice(eta_range, size=10)
beta1s = np.random.choice(beta1_range, size=10)
beta2s = np.random.choice(beta2_range, size=10)


def write_config_files(etas, beta1s, beta2s, folder):
    """"""
    folder = folder + "/"
    N = len(etas)
    for i in range(N):
        with open(folder + "config" + str(i) + '.txt', 'w') as f:
            f.write(f"eta = {etas[i]:}\n")
            f.write(f"beta1 = {beta1s[i]:}\n")
            f.write(f"beta2 = {beta2s[i]:}\n")
            f.write(f"lambda = 0.001\n")
            f.write(f"epochs = 100\n")
            f.write(f"batchsize = 5000\n")
            f.write(f"loss_function = MAE")

        with open(folder + "job" + str(i) + '.txt', 'w') as f:
            text = f"""#!/usr/local_rwth/bin/zsh
#SBATCH --output job{i:}.out
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=4
#SBATCH --job-name=model{i:}

cd $HOME/NewtonNet/Project2
### Load python module
module load python/3.8.7

### Execute your application
python3 train_model.py -f $WORK/NewtonNet/{folder:}config{i:}.txt -o $WORK/NewtonNet/{folder:}model{i:}"""
            f.write(text)


folder = "test"
write_config_files(etas, beta1s, beta2s, folder)
