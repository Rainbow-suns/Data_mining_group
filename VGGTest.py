from Common_Operation import test, initial_test


def VGGTest():
    test_dataloader, model = initial_test('VGGnet', 10)
    accuracy = test(test_dataloader, model)

    return accuracy


if __name__ == '__main__':
    VGGTest()
