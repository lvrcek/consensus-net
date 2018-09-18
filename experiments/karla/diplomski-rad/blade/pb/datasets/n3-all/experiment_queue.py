import os

if __name__ == '__main__':
    experiments = [f for f in sorted(os.listdir('./'))
                   if f.startswith('model-testing')]
    for experiment in experiments:
        os.system('python3 {}'.format(experiment))
