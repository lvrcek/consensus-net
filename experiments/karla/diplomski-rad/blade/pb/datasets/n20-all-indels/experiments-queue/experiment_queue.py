import os

if __name__ == '__main__':
    experiments = [f for f in sorted(os.listdir('./'))
                   if f.startswith('model')]
    print(experiments)
    for experiment in experiments:
        os.system('python3 {}'.format(experiment))
