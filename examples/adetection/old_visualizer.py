def old():
    for i in range(0, dataset.test_data.shape[0]):  # range( dataset.test_data.shape[0]):
        cloud = dataset.test_data[i]

        temp = np.copy(cloud[:, 1])
        cloud[:, 1] = cloud[:, 2]
        cloud[:, 2] = temp

        dist_string = "{:.5f}".format(scores[i])
        # "~/Documenti/thesis/results/AD/"
        path = "/home/Alberto/" + "name" + ".png"
        v = pptk.viewer(cloud, cloud[:, 2])
        v.set(point_size=0.010)
        v.set(phi=3.14 / 6)
        v.set(theta=3.14 / 6)
        v.set(r=5)
        v.color_map("winter")
        v.set(show_grid=False)
        v.set(bg_color=[1, 1, 1, 1])
        v.set(show_axis=False)
        time.sleep(5)

        if dataset.test_labels[i]:
            label = "N"
        else:
            label = "A"
        name1 = "dist-" + dist_string + "-img-" + str(i) + "-" + label + ".png"
        # im = pyautogui.screenshot(name1, region=(40, 96, 510, 510))
        print(name1)  # 40, 96 ... 550, 608
        v.close()
        time.sleep(3)

