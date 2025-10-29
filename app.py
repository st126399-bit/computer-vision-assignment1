def main():
    print("=== Computer Vision App ===")
    print("1. Color Conversion")
    print("2. Adjust Brightness/Contrast")
    print("3. Show Histogram")
    print("4. Gaussian Filter")
    print("5. Bilateral Filter")
    print("6. Canny Edge Detection")
    print("7. Hough Line Detection")
    print("8. Panorama")
    print("9. Transform Image")
    print("10. Calibrate Camera")
    print("11. Augmented Reality")
    choice = int(input("Select option: "))

    img = cv2.imread('test.jpg')

    if choice == 1:
        out = convert_color(img, 'hsv')
    elif choice == 2:
        out = adjust_contrast_brightness(img, 1.5, 40)
    elif choice == 3:
        show_histogram(img)
        return
    elif choice == 4:
        out = gaussian_filter(img)
    elif choice == 5:
        out = bilateral_filter(img)
    elif choice == 6:
        out = canny_edge(img)
    elif choice == 7:
        out = hough_lines(img)
    elif choice == 8:
        img2 = cv2.imread('test2.jpg')
        out = create_panorama(img, img2)
    elif choice == 9:
        out = transform_image(img)
    elif choice == 10:
        import calibration_images  # your existing script
        return
    elif choice == 11:
        import ar_aruco_overlay_save  # your AR code
        return
    else:
        print("Invalid option")
        return

    cv2.imshow("Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
