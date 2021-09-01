import os
import subprocess

PLY_PATH = "../data/shapenet_ply"
GOOD_PATH = "../data/shapenet_GOOD_5"

TRAIN_PLYS = os.path.join(PLY_PATH, "train")
TEST_PLYS = os.path.join(PLY_PATH, "test")
TRAIN_GOOD = os.path.join(GOOD_PATH, "train")
TEST_GOOD = os.path.join(GOOD_PATH, "test")

if __name__ == "__main__":
    print("\n\nstarting to compute descriptors for Train data...\n\n")
    for cls in os.listdir(TRAIN_PLYS):
        save_dir = os.path.join(TRAIN_GOOD, cls)
        read_dir = os.path.join(TRAIN_PLYS, cls)
        print(read_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for pc in os.listdir(read_dir):
            input_pc = os.path.join(read_dir, pc)
            output_pc = os.path.join(save_dir, pc)
            print(input_pc)
            ret = subprocess.run(["./test_GOOD_descriptor", input_pc, output_pc+".good","5"], capture_output=False)
            if ret.returncode != 0:
                raise Exception
            print("saved GOOD descriptor for" + input_pc)
            print("saving in " + output_pc)

    print("\n\nstarting to compute descriptors for test data...\n\n")
    for cls in os.listdir(TEST_PLYS):
        save_dir = os.path.join(TEST_GOOD, cls)
        read_dir = os.path.join(TEST_PLYS, cls)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for pc in os.listdir(read_dir):
            input_pc = os.path.join(read_dir, pc)
            output_pc = os.path.join(save_dir, pc)
            ret = subprocess.run(["./test_GOOD_descriptor", input_pc, output_pc+".good", "5"])
            if ret.returncode != 0:
                raise Exception
            print("saved GOOD descriptor for" + input_pc)
            print("saving in " + output_pc)
