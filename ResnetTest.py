from Common_Operation import test, initial_test


def ResnetTest():
    test_dataloader, model = initial_test('Resnet', 100)
    accuracy = test(test_dataloader, model)

    return accuracy


if __name__ == '__main__':
    ResnetTest()
