import shutil

from ResnetTest import ResnetTest
from ResnetTrain import ResnetTrain
from VGGTest import VGGTest
from VGGTrain import VGGTrain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm_type', default=1, type=int)

args = parser.parse_args()

save_path = './Best_model'
algorithms = ['Resnet', 'VGGnet']
algorithm_type = args.algorithm_type
algorithm = algorithms[algorithm_type]
path_1 = './TrainedModel/' + algorithm + '.pth'

best_result = 0.0
accuracy = 0.0

for lr in [8e-5, 9e-5, 1e-4, 2e-4, 3e-4]:
    for bs in [4, 5, 6, 7, 8]:
        if algorithm_type == 0:
            ResnetTrain(lr, bs)
            accuracy = ResnetTest()
        elif algorithm_type == 1:
            VGGTrain(lr, bs)
            accuracy = VGGTest()

        if accuracy > best_result:  # Find the best performing parameter
            best_result = accuracy
        shutil.copy(path_1, save_path)
        best_parameters = {'lr': lr, 'bs': bs}

print("Best result:{:.3f}".format(best_result))
print("Best parameters:{}".format(best_parameters))