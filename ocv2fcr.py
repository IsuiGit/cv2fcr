import cv2, random, string
from faces.faces import loadFaces, saveFaces

class cv2fcr:
    def __init__(self):
        self.stream = None
        self.net = cv2.dnn.readNet("data/ocv2fcr.pb", "data/ocv2fcr.pbtxt")
        self.faces = self.cv2fcrLoadFaces()

    def cv2fcrLoadFaces(self):
        data, e = loadFaces()
        if e:
            print(repr(e))
        else:
            return data

    def cv2fcrUpdateFaces(self):
        self.faces = self.cv2fcrLoadFaces()

    def cv2FaceRecognition(self, frame, tresh=0.7):
        try:
            # Получаем высоту и ширину
            frameData = [frame.shape[0], frame.shape[1]]
            # Блобим в двоичный объект
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
            # Подаъем сети на вход объект
            self.net.setInput(blob)
            # Прямой проход для распознавания
            detections = self.net.forward()
            # Создаём рамки вокруг лица
            faceBoxes=[]
            # Прямой перебор всех результатов
            for i in range(detections.shape[2]):
                # получаем результат вычислений для очередного элемента
                confidence = detections[0,0,i,2]
                # если результат превышает порог срабатывания — это лицо
                if confidence > tresh:
                    # формируем координаты рамки
                    x1=int(detections[0,0,i,3]*frameData[1])
                    y1=int(detections[0,0,i,4]*frameData[0])
                    x2=int(detections[0,0,i,5]*frameData[1])
                    y2=int(detections[0,0,i,6]*frameData[0])
                    # добавляем их в общую переменную
                    faceBoxes.append([x1,y1,x2,y2])
                    # рисуем рамку на кадре
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), int(round(frameData[0]/150)), 8)
                    cropedFrame = frame[y1:y2, x1:x2]
                    # возвращаем кадр с рамками
                    return frame, cropedFrame, faceBoxes, None
                else:
                    return frame, None, None, None
        except Exception as e:
            return None, None, None, repr(e)

    def cv2FaceCollections(self, frame):
        try:
            if self.faces:
                id = None
                for i in self.faces.keys():
                    proto = cv2.imread(self.faces[i]['proto'])
                    height, width = proto.shape[:2]
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    if cv2.matchTemplate(frame, proto, cv2.TM_CCOEFF_NORMED).max() > 0.75:
                        id = i
                    elif cv2.matchTemplate(frame, proto, cv2.TM_CCOEFF_NORMED).max() < 0.05:
                        id = False
                    else:
                        pass
                if id == False:
                    id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
                    cv2.imwrite(f'faces/img/{id}.jpg', frame)
                    data = {'shape': frame.shape, 'proto': f'faces/img/{id}.jpg', 'name': ''}
                    self.faces[id] = data
                    saveFaces(self.faces)
                    print('New undefined person!')
                    return self.faces[id]['name'], None
                elif id == None:
                    return None, None
                else:
                    return self.faces[id]['name'], None

            else:
                id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
                cv2.imwrite(f'faces/img/{id}.jpg', frame)
                data = {'shape': frame.shape, 'proto': f'faces/img/{id}.jpg', 'name': ''}
                self.faces[id] = data
                saveFaces(self.faces)
                print('New undefined person!')
                return self.faces[id]['name'], None
        except Exception as e:
            return None, repr(e)

    def cv2StreamFCR(self):
        self.stream = cv2.VideoCapture(0)
        while cv2.waitKey(1)<0:
            # получаем очередной кадр с камеры
            hasFrame, frame = self.stream.read()
            # если кадра нет
            if not hasFrame:
                # останавливаемся и выходим из цикла
                cv2.waitKey()
                break
            # распознаём лица в кадре
            resultImg, face, faceBoxes, e = self.cv2FaceRecognition(frame)
            if e:
                print(e)
                break
            else:
                # Если лицо есть
                if faceBoxes:
                    res, e = self.cv2FaceCollections(face)
                    self.cv2fcrUpdateFaces()
                    if e:
                        print(e)
                    else:
                        if res != None:
                            res = res.encode('cp1251').decode('utf-8')
                            cv2.putText(resultImg, res, (faceBoxes[0][0], faceBoxes[0][1]), cv2.FONT_HERSHEY_COMPLEX , .7, (0, 0, 0), 2, cv2.LINE_AA)
                        else:
                            pass
                else:
                    pass
                # выводим картинку с камеры
                cv2.imshow("Face detection", resultImg)

cv2fcrDaemon = cv2fcr()
cv2fcrDaemon.cv2StreamFCR()
