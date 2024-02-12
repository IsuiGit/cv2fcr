import ocv2fcr, cv2, keyboard, io
import PySimpleGUI as sg
from PIL import Image

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
                    select_mode = 'extended'
                ),
            ],
            [sg.Button("Merge choosen"), sg.Button("Edit")]
        ]
        merge_window = sg.Window('Detections merge', layout, resizable=True, finalize=True)
        keyboard.get_hotkey_name()
        multiple = []
        while True:
            event, values = merge_window.read(timeout=20)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            elif event == '-MERGE_TABLE-':
                if keyboard.is_pressed('ctrl'):
                    multiple = []
                    index = values['-MERGE_TABLE-']
                    for i in index:
                        multiple.append(self.daemon.t_faces[i])
                else:
                    try:
                        index = values['-MERGE_TABLE-'][0]
                        multiple = [self.daemon.t_faces[index]]
                    except:
                        pass
            elif event == 'Merge choosen':
                if not multiple:
                    print('No choosed person')
                else:
                    self.daemon.cv2MergePersons(multiple)
                    merge_window['-MERGE_TABLE-'].update(values=self.daemon.t_faces)
            elif event == 'Edit':
                self.edit(self.daemon.t_faces[values['-MERGE_TABLE-'][0]])

    def edit(self, data):
        layout = [
            [sg.Image(filename='', key='-IMAGE-')],
            [sg.Text(data[0], expand_x=True, justification='center')],
            [sg.Input(data[3], enable_events=True, key='-INPUT-', expand_x=True, justification='left')],
            [sg.Button("Save", size=(10, 1))],
        ]
        edit_window = sg.Window(f'Edit {data[0]}', layout, resizable=True, finalize=True)
        image = Image.open(data[2])
        image.thumbnail((400, 400))
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        edit_window["-IMAGE-"].update(data=bio.getvalue())
        while True:
            event, values = edit_window.read(timeout=20)
            if event == 'Exit' or event == sg.WIN_CLOSED:
                break
            elif event == 'Save':
                self.daemon.faces[data[0]]['name'] = values['-INPUT-'].encode('utf-8').decode('cp1251')
                res = self.daemon.cv2fcrSaveFaces()
                if res == True:
                    sg.popup_no_buttons('Changes saved!', non_blocking=True)
                else:
                    sg.popup_no_buttons(res, non_blocking=True)

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
            elif event == "Merge" and self.daemon.faces:
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
