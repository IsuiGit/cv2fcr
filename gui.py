import ocv2fcr, cv2
import PySimpleGUI as sg

class cv2fcrGUI:
    def __init__(self):
        self.daemon = ocv2fcr.cv2fcr()
        self.stream = cv2.VideoCapture(0)
        sg.theme("GrayGrayGray")

    def merge(self):
        headings = ['ID', 'Shape', 'Proto', 'Name', 'Similar']
        layout = [
            [
                sg.Table(
                    values=self.daemon.t_faces,
                    headings = headings,
                    max_col_width=35,
                    auto_size_columns=True,
                    display_row_numbers=True,
                    justification='c',
                    num_rows=10,
                    enable_events = True,
                    key = '-MERGE_TABLE-',
                    row_height = 50,
                ),
            ],
        ]

        merge_window = sg.Window('Detections merge', layout, resizable=True, finalize=True)
        while True:
            event, values = merge_window.read(timeout=20)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            elif event == '-MERGE_TABLE-':
                index = values['-MERGE_TABLE-'][0]
                print(self.daemon.t_faces[index])

    def main(self):
        layout = [
            [sg.Image(filename="", key="-IMAGE-")],
            [sg.Button("Merge", size=(10, 1)), sg.Button("Exit", size=(10, 1))],
        ]
        # Create the window and show it without the plot
        window = sg.Window("OpenCV Face Recognition", layout, location=(800, 400))
        count = 0
        while True:
            event, values = window.read(timeout=20)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            elif event == "Merge":
                self.merge()
            hasFrame, frame = self.stream.read()
            # если кадра нет
            if not hasFrame:
                # останавливаемся и выходим из цикла
                cv2.waitKey()
                break
            count += 1
            resultImg, face, faceBoxes, e = self.daemon.cv2FaceRecognition(frame)
            if e:
                print(e)
                break
            else:
                # Если лицо есть
                if faceBoxes:
                    if count%10 == 0:
                        res, e = self.daemon.cv2FaceCollect(face)
                        self.daemon.cv2fcrUpdateFaces()
                    else:
                        res = None
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
                imgbytes = cv2.imencode(".png", resultImg)[1].tobytes()
                window["-IMAGE-"].update(data=imgbytes)
        window.close()

if __name__ == '__main__':
    gui = cv2fcrGUI()
    gui.main()
