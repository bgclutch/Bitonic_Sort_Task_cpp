import random
import os

path = "input_files/"
os.makedirs(path, exist_ok=True)

tests_number = 10

for test_number in range(0, tests_number):
    name_of_file = path + "test_" + f'{test_number + 1:02}' + ".in"
    file = open(name_of_file, 'w')
    test_text = ""
    test_data_size = random.randint(8, 8)
    test_text += str(test_data_size) + " "
    for operation_number in range (0, test_data_size):
        test_text += str(random.randint(-2000000, 2000000)) + " "
    file.write(test_text)
    file.close()