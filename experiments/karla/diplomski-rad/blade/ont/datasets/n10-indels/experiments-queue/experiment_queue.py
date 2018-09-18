import os

if __name__ == '__main__':
    experiments = [f for f in sorted(os.listdir('./'))
                   if f.startswith('model-n10-indel-ont-1.py')]
    for experiment in experiments:
        os.system('python3 {}'.format(experiment))
