import os
import face_recognition
import cv2
import numpy
import threading
import numpy as np
from sklearn.metrics import precision_score


# class that inherits from thread. it creates a separate thread for showing cv2 images.
class custom_thread(threading.Thread):

    # c'tor
    # @params: imgList: images to show.
    # @params: txtList: names of the people in the images in correspond to the indexes of it.
    def __init__(self, imgList, txtList):
        threading.Thread.__init__(self)
        self.imgList = imgList
        self.txtList = txtList

    # run- show the selected images with the name that recognized.
    def run(self):
        # variable that tells to the main thread that there are still open windows of images.
        global windowsClosed
        windowsClosed = False

        # show images from imgList with texts from txtList
        index = 0
        while index < len(self.imgList):
            cv2.putText(self.imgList[index], f'{self.txtList[index]}', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                        (0, 0, 255), 2)
            cv2.imshow(f'{str(index + 1)}. {self.txtList[index]}', self.imgList[index])
            index += 1

        # wait to close all the opened windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        windowsClosed = True


windowsClosed = True

print("NOTE: Only images in JPG format are accepted.\nInput images at InputImages.\nDatabase at DBImages.\nImages to add into database at DBImagesToAdd.\n")
choose = -1
print("Welcome!")
while True:
    m = input("---\nChoose:\n"
              "\t1. HOG.\n"
              "\t2. HAAR CASCADE combined with LBPH.\n"
              "\t3. Exit the program.\n"
              "---\n")

    if m == '1':
        while True:
            choose = input("---\nChoose:\n"
                           "\t1. Scan the database and save the data.\n"
                           "\t2. Use saved data and show result.\n"
                           "\t3. Add data from folder DBImagesToAdd to saved data.\n"
                           "\t4. Calculate precision for HOG.\n"
                           "\t5. Exit HOG.\n"
                           "---\n")
            if choose == '1':
                try:
                    path = 'DBImages'
                    if not os.path.exists(path):
                        print('Database folder "DBImages" does not exist. Create it and try again')
                        continue
                    encodingsList = []  # list of image encoded.
                    personNamesList = []  # list of persons names corresponding to the encodings of their images.
                    print("Scanning database, please wait...")

                    # for each directory in directory DBImages
                    for directory in os.listdir(path):
                        personName = str(directory)  # name of directory
                        personPath = os.path.join(path, directory)  # path to the directory with images of the person

                        # --HOG METHOD--
                        # for each image in person directory
                        for img in os.listdir(personPath):
                            if os.path.splitext(img)[1] != ".jpg":
                                continue
                            img2 = face_recognition.load_image_file(
                                os.path.join(personPath, img))  # load the person's image
                            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                            try:
                                encodeImg2 = face_recognition.face_encodings(img2)[0]  # gets only first recognized face
                                encodingsList.append(encodeImg2)
                                personNamesList.append(personName)

                            except IndexError:
                                pass

                    # saving results
                    npArray = numpy.array(encodingsList)
                    numpy.savetxt("encodings.txt", npArray)

                    npArray = numpy.array(personNamesList)
                    numpy.savetxt("personNamesList.txt", npArray, fmt="%s")
                    print("Result saved successfully")
                except Exception as e:
                    print("Error occurred while scanning saved data, it may be corrupted.\nFull error:")
                    print(e)
            elif choose == '2':
                if not windowsClosed:
                    print('Close all windows with images and then try again')
                    continue
                try:
                    imagesToShow = []
                    namesToShow = []
                    if not os.path.exists("InputImages"):
                        print('Input folder "InputImages" does not exist. Create it and try again')
                        continue
                    print("Processing results, please wait...")

                    encodingsList = numpy.loadtxt('encodings.txt')
                    personNamesList = numpy.loadtxt('personNamesList.txt', dtype='str')

                    for img in os.listdir("InputImages"):
                        if os.path.splitext(img)[1] != ".jpg":
                            continue
                        img1 = face_recognition.load_image_file(os.path.join('InputImages', str(img)))
                        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                        encodeImg1 = face_recognition.face_encodings(img1)[0]
                        results = face_recognition.compare_faces(encodingsList, encodeImg1)
                        faceDis = face_recognition.face_distance(encodingsList, encodeImg1)
                        i = 0
                        closestResult = -1
                        while i < len(results):
                            result = results[i]
                            if result:
                                if closestResult == -1:
                                    closestResult = i
                                elif faceDis[i] < faceDis[closestResult]:
                                    closestResult = i
                            i += 1
                        if closestResult != -1:
                            namesToShow.append(personNamesList[closestResult])
                        else:
                            namesToShow.append('Image not recognized')
                        imagesToShow.append(img1)

                    thread = custom_thread(imagesToShow, namesToShow)
                    thread.start()

                except FileNotFoundError:
                    print('No saved data, scan the database')
                except Exception as e:
                    print("Error occurred while scanning saved data, it may be corrupted.\nFull error:")
                    print(e)
            elif choose == '3':
                try:
                    path = 'DBImagesToAdd'
                    if not os.path.exists(path):
                        print('Folder "DBImagesToAdd" does not exist. Create it and try again')
                        continue
                    try:
                        encodingsList = []
                        personNamesList = []
                        print("Scanning images, please wait...")
                        for directory in os.listdir(path):
                            personName = str(directory)
                            personPath = os.path.join(path, directory)
                            for img in os.listdir(personPath):
                                if os.path.splitext(img)[1] != ".jpg":
                                    continue
                                img2 = face_recognition.load_image_file(os.path.join(personPath, img))
                                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                                try:
                                    encodeImg2 = face_recognition.face_encodings(img2)[0]  # gets only first recognized face
                                    encodingsList.append(encodeImg2)
                                    personNamesList.append(personName)
                                except IndexError:
                                    pass
                        # saving results
                        npArray = numpy.array(encodingsList)
                        with open("encodings.txt", "ab") as f:
                            numpy.savetxt(f, npArray)
                        npArray = numpy.array(personNamesList)
                        with open("../../Desktop/personNamesList.txt", "ab") as f:
                            numpy.savetxt(f, npArray, fmt="%s")
                        print("Result saved successfully")
                    except FileNotFoundError:
                        print('No saved data, scan the database')
                    except Exception as e:
                        print("Error occurred while scanning saved data, it may be corrupted.\nFull error:")
                        print(e)
                except Exception as e:
                    print("Error occurred while scanning saved data, it may be corrupted.\nFull error:")
                    print(e)
            elif choose == '4':
                if not windowsClosed:
                    print('Close all windows with images and then try again')
                    continue
                try:
                    imagesToShow = []
                    namesToShow = []
                    if not os.path.exists("InputImages"):
                        print('Input folder "InputImages" does not exist. Create it and try again')
                        continue
                    print("Processing results, please wait...")

                    encodingsList = numpy.loadtxt('encodings.txt')
                    personNamesList = numpy.loadtxt('personNamesList.txt', dtype='str')
                    realNames = []
                    for img in os.listdir("InputImages"):
                        if os.path.splitext(img)[1] != ".jpg":
                            continue
                        imgRealName = input(f'Enter real name for image "{img}":')
                        realNames.append(imgRealName)
                        img1 = face_recognition.load_image_file(os.path.join('InputImages', str(img)))
                        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                        encodeImg1 = face_recognition.face_encodings(img1)[0]
                        results = face_recognition.compare_faces(encodingsList, encodeImg1)
                        faceDis = face_recognition.face_distance(encodingsList, encodeImg1)
                        i = 0
                        closestResult = -1
                        while i < len(results):
                            result = results[i]
                            if result:
                                if closestResult == -1:
                                    closestResult = i
                                elif faceDis[i] < faceDis[closestResult]:
                                    closestResult = i
                            i += 1
                        if closestResult != -1:
                            namesToShow.append(personNamesList[closestResult])
                        else:
                            namesToShow.append('Image not recognized')
                        imagesToShow.append(img1)
                    print(f'Precision: {precision_score(realNames, namesToShow, average="micro")}')   # THIS IS THE PRECISION SCORE. HOW MANY PICTURES WAS RECOGNIZED CORRECTLY DIVIDED BY NUMBER OF PICTURES
                    thread = custom_thread(imagesToShow, namesToShow)
                    thread.start()
                except FileNotFoundError:
                    print('No saved data, scan the database')
                except Exception as e:
                    print("Error occurred while scanning saved data, it may be corrupted.\nFull error:")
                    print(e)
            elif choose == '5':
                if not windowsClosed:
                    print('Close all windows with images and then try again')
                    continue
                break
            else:
                print("Invalid choice, try again")
    elif m == '2':
        while True:
            choose = input("---\nChoose:\n"
                           "\t1. Scan the database and save the data (Train the system).\n"
                           "\t2. Use saved data from the DB and show result.\n"
                           "\t3. Calculate precision for HAARCASCADE.\n"
                           "\t4. Exit HAARCASCADE.\n"
                           "---\n")
            if choose == '1':
                try:
                    people = []

                    DIR = "DBImages"
                    if not os.path.exists(DIR):
                        print('Folder "DBImages" does not exist. Create it and try again')
                        continue
                    #
                    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

                    # haar cascade
                    features = []

                    # The images array of faces.
                    labels = []

                    # Every face in this features list, what is corresponding label, whose face does it belong to.
                    print("Scanning database, please wait...")
                    # read from file - the name of the people which is the name of the files
                    for directory in os.listdir(DIR):
                        k = os.path.join(DIR, directory)
                        people.append(str(directory))
                    for person in people:
                        # check if get stack when we possess the image
                        path = os.path.join(DIR, person)
                        label = people.index(person)

                        # Loop over every person in the people list, garb the path for this person.

                        # take each img from the file of the person
                        for img in os.listdir(path):
                            if os.path.splitext(img)[1] != ".jpg":
                                continue

                            # take the full path of the img
                            img_path = os.path.join(path, img)

                            # we are inside each folder, we will loop over every image in that folder.

                            # goes to the picture, and read it- make an array of integers that represent it
                            img_array = cv2.imread(img_path)

                            # make another picture out of the picture in grey
                            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                            # Now we have path of image, we will read these image, and turn color from blue-green-red to

                            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                            # Detect the faces with the help of haar cascade

                            for (x, y, w, h) in faces_rect:
                                faces_roi = gray[y:y + h, x:x + w]

                                features.append(faces_roi)
                                labels.append(label)
                                break #to detect only first face
                    # Loop over every face in faces_rect and append features and labels

                    # features = np.array(features, dtype='object')
                    labels = np.array(labels)

                    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

                    # Train the Recognizer on the features lost and the labels list.

                    # Attention, please load opencv_contrib_python to find face function in opencv library

                    # first array needs to be regular, the second is np array.
                    face_recognizer.train(features, labels)

                    # Convert features and labels to numpy array.
                    if os.path.isfile("face_trained.yml"):
                        os.remove("face_trained.yml")
                    face_recognizer.save('face_trained.yml')
                    people = np.array(people)
                    numpy.savetxt("peopleNamesHAAR.txt", people, fmt="%s")
                    #numpy.savetxt("personNamesListHAAR.txt", labels, fmt="%s")
                    print("Result saved successfully")
                except Exception as e:
                    print("Error occurred while scanning saved data, it may be corrupted.\nFull error:")
                    print(e)
            elif choose == '2':
                try:
                    if not windowsClosed:
                        print('Close all windows with images and then try again')
                        continue
                    # load the classifier
                    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    imagesList = []
                    textsList = []
                    people = []
                    size = (250, 250)
                    DIR = "InputImages"
                    if not os.path.exists(DIR):
                        print('Input folder "InputImages" does not exist. Create it and try again')
                        continue
                    people = numpy.loadtxt('peopleNamesHAAR.txt', dtype='str')

                    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

                    face_recognizer.read('face_trained.yml')

                    for img in os.listdir("InputImages"):
                        if os.path.splitext(img)[1] != ".jpg":
                            continue
                        img1 = cv2.imread(os.path.join(DIR, img))
                        imagesList.append(img1)
                        img1 = cv2.resize(img1, size)

                        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
                        for (x, y, w, h) in faces_rect:
                            faces_roi = gray[y:y + h, x:x + w]
                            label, confidence = face_recognizer.predict(faces_roi)
                            textsList.append(str(people[label]))
                            break
                    thread = custom_thread(imagesList, textsList)
                    thread.start()
                except Exception as e:
                    print("Error occurred while scanning saved data, it may be corrupted.\nFull error:")
                    print(e)
            elif choose == '3':
                try:
                    if not windowsClosed:
                        print('Close all windows with images and then try again')
                        continue
                    # load the classifier
                    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    imagesList = []
                    textsList = []
                    people = []
                    size = (250, 250)
                    DIR = "InputImages"
                    if not os.path.exists(DIR):
                        print('Input folder "InputImages" does not exist. Create it and try again')
                        continue
                    people = numpy.loadtxt('peopleNamesHAAR.txt', dtype='str')

                    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

                    face_recognizer.read('face_trained.yml')
                    realNames = []
                    for img in os.listdir("InputImages"):
                        if os.path.splitext(img)[1] != ".jpg":
                            continue
                        imgRealName = input(f'Enter real name for image "{img}":')
                        realNames.append(imgRealName)
                        img1 = cv2.imread(os.path.join(DIR, img))
                        imagesList.append(img1)
                        img1 = cv2.resize(img1, size)

                        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
                        for (x, y, w, h) in faces_rect:
                            faces_roi = gray[y:y + h, x:x + w]
                            label, confidence = face_recognizer.predict(faces_roi)
                            textsList.append(str(people[label]))
                            break
                    print(f'Precision: {precision_score(realNames, textsList, average="micro")}')   # THIS IS THE PRECISION SCORE. HOW MANY PICTURES WAS RECOGNIZED CORRECTLY DIVIDED BY NUMBER OF PICTURES
                    thread = custom_thread(imagesList, textsList)
                    thread.start()
                except Exception as e:
                    print("Error occurred while scanning saved data, it may be corrupted.\nFull error:")
                    print(e)
            elif choose == '4':
                if not windowsClosed:
                    print('Close all windows with images and then try again')
                    continue
                break
    elif m == '3':
        if not windowsClosed:
            print('Close all windows with images and then try again')
            continue
        break
    else:
        print("Invalid choice, try again")