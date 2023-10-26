'''
OpenCV (Open Source Computer Vision) is a library with functions that mainly aiming real-time computer vision.
OpenCV supports Deep Learning frameworks Caffe which has been implemented in this project.
With OpenCV we have perform face detection using pre-trained
deep learning face detector model which is shipped with the library.
'''
# Import the necessary packages
import cv2
import numpy as np
import math
import argparse
from flask import Flask, render_template, Response, request
from PIL import Image
import io

UPLOAD_FOLDER = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def highlightFace(net, frame, conf_threshold=0.7):
    # Tạo một bản sao của khung hình gốc
    frameOpencvDnn = frame.copy()
    
    # Lấy chiều cao và chiều rộng của khung hình
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    # Tạo một "blob" từ khung hình sử dụng OpenCV
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    
    # Đưa blob vào mạng neural để thực hiện dự đoán
    net.setInput(blob)
    
    # Lấy các dự đoán và phát hiện
    detections = net.forward()

    # Danh sách để lưu trữ tọa độ của các khuôn mặt
    faceBoxes = []

    # Duyệt qua các dự đoán để xác định và vẽ hộp bao quanh khuôn mặt
    for i in range(detections.shape[2]):
        # Lấy tỷ lệ tự tin (confidence) của dự đoán
        confidence = detections[0, 0, i, 2]
        
        # So sánh tỷ lệ tự tin với ngưỡng (threshold) được đưa vào
        if confidence > conf_threshold:
            # Tính toán tọa độ của hộp bao quanh khuôn mặt
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            
            # Thêm tọa độ vào danh sách
            faceBoxes.append([x1, y1, x2, y2])
            
            # Vẽ hộp bao quanh khuôn mặt lên khung hình
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    # Trả về khung hình đã được vẽ hộp bao quanh khuôn mặt và danh sách tọa độ của khuôn mặt
    return frameOpencvDnn, faceBoxes



# đặt ảnh đầu vào cho chương trình để thực hiện quá trình phát hiện
# mã sử dụng thư viện argparse
parser = argparse.ArgumentParser()
# khởi tạo một đối tượng ArgumentParser từ thư viện argparse
# nếu không có tham số đầu vào được cung cấp, chương trình sẽ bỏ qua dòng parser.add_argument('--image')
#  và sử dụng webcam để thực hiện quá trình phát hiện.
parser.add_argument('--image')

args = parser.parse_args()

'''
Mỗi mô hình đi kèm với hai tệp: tệp trọng lượng (weight file) và tệp mô hình (model file).
Tệp trọng lượng lưu trữ dữ liệu về triển khai của mô hình, trong đó bao gồm trọng số của các lớp và tham số khác cần cho việc dự đoán.
Tệp mô hình lưu trữ dự đoán thực tế được thực hiện bởi mô hình, có nghĩa là kết quả của việc đào tạo mô hình trên dữ liệu đào tạo.
Chúng ta sử dụng các mô hình được đào tạo trước.
Các tệp .prototxt là các tệp định nghĩa kiến trúc mô hình (nghĩa là, các lớp trong mô hình). Tệp này mô tả cách các lớp được kết nối với nhau và các tham số cấu hình cho mỗi lớp.
Tệp .caffemodel chứa trọng số thực tế cho các lớp trong mô hình. Các trọng số này là kết quả của việc đào tạo mô hình trên dữ liệu thực tế và chứa thông tin cụ thể để thực hiện các dự đoán.
Cả hai tệp đều cần thiết khi sử dụng các mô hình được đào tạo bằng Caffe cho học sâu (deep learning).
'''

def gen_frames():
    faceProto = "opencv_face_detector.pbtxt"  # Tệp prototxt cho mô hình nhận dạng khuôn mặt
    faceModel = "opencv_face_detector_uint8.pb"  # Tệp caffeModel cho mô hình nhận dạng khuôn mặt
    ageProto = "age_deploy.prototxt"  # Tệp prototxt cho mô hình nhận dạng tuổi tác
    ageModel = "age_net.caffemodel"  # Tệp caffeModel cho mô hình nhận dạng tuổi tác
    genderProto = "gender_deploy.prototxt"  # Tệp prototxt cho mô hình nhận dạng giới tính
    genderModel = "gender_net.caffemodel"  # Tệp caffeModel cho mô hình nhận dạng giới tính

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # Giá trị trung bình cho mô hình
    # Danh sách phạm vi tuổi tác
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']  # Danh sách giới tính

    # LOAD NETWORK (Nạp mạng neural)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)  # Đọc mô hình nhận dạng khuôn mặt
    ageNet = cv2.dnn.readNet(ageModel, ageProto)  # Đọc mô hình nhận dạng tuổi tác
    genderNet = cv2.dnn.readNet(genderModel, genderProto)  # Đọc mô hình nhận dạng giới tính

    # Mở video từ camera (ID = 0)
    video = cv2.VideoCapture(0)

    # Vùng đệm (padding) cho khuôn mặt
    padding = 20

    while cv2.waitKey(1) < 0:
        # Đọc khung hình từ video
        hasFrame, frame = video.read()

        if not hasFrame:
            cv2.waitKey()
            break

        # Nhận dạng khuôn mặt trong khung hình
        resultImg, faceBoxes = highlightFace(faceNet, frame)

        if not faceBoxes:
            print("No face detected")  # Nếu không có khuôn mặt nào được phát hiện

        for faceBox in faceBoxes:
            # Cắt khung hình khuôn mặt từ khung hình gốc
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - padding):
                         min(faceBox[2] + padding, frame.shape[1] - 1)]

            # Tạo một blob từ hình ảnh khuôn mặt để chuẩn bị cho việc nhận dạng giới tính và tuổi tác
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)

            # Dự đoán giới tính của khuôn mặt
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            print(f'Gender: {gender}')  # In giới tính lên console

            ageNet.setInput(blob)

            # Dự đoán tuổi tác của khuôn mặt
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            print(f'Age: {age[1:-1]} years')  # In tuổi tác lên console

            # Hiển thị văn bản trên hình ảnh với thông tin giới tính và tuổi tác
            cv2.putText(resultImg, f'{gender}, {age}',
                        (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            if resultImg is None:
                continue

            # Chuyển đổi hình ảnh kết quả sang định dạng JPEG và gửi nó đi
            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')



def gen_frames_photo(img_file):
    faceProto = "opencv_face_detector.pbtxt"  # Tệp prototxt cho mô hình nhận dạng khuôn mặt
    faceModel = "opencv_face_detector_uint8.pb"  # Tệp caffeModel cho mô hình nhận dạng khuôn mặt
    ageProto = "age_deploy.prototxt"  # Tệp prototxt cho mô hình nhận dạng tuổi tác
    ageModel = "age_net.caffemodel"  # Tệp caffeModel cho mô hình nhận dạng tuổi tác
    genderProto = "gender_deploy.prototxt"  # Tệp prototxt cho mô hình nhận dạng giới tính
    genderModel = "gender_net.caffemodel"  # Tệp caffeModel cho mô hình nhận dạng giới tính

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # Danh sách phạm vi tuổi tác
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
    genderList = ['Male', 'Female'] # Danh sách giới tính

    # LOAD NETWORK (Nạp mạng Neural)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)  # Đọc mô hình nhận dạng khuôn mặt
    ageNet = cv2.dnn.readNet(ageModel, ageProto)  # Đọc mô hình nhận dạng tuổi tác
    genderNet = cv2.dnn.readNet(genderModel, genderProto)  # Đọc mô hình nhận dạng giới tính

# Mở 1 video hoặc 1 bức ảnh hoặc camera trực tiếp

    frame = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
    #img_file: biến đại diện cho hình ảnh đầu vào
    #cv2.COLOR_BGR2RGB: chỉ định chuyển đổi từ không gian màu BGR (Blue-Green-Red) sang không gian màu RGB (Red-Green-Blue)
    #cv2.cvtColor(): Đây là một hàm trong thư viện OpenCV để thực hiện phép chuyển đổi không gian màu của hình ảnh

    # Vùng đệm (padding) cho khuôn mặt
    padding = 20
    while cv2.waitKey(1) < 0:
        # Đọc khung hình từ video

        # Nhận dạng từ khuôn mặt trong hình
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:   # Không phát hiện khuôn mặt
            print("No face detected")   

       for faceBox in faceBoxes:
            # Cắt khung hình khuôn mặt từ khung hình gốc
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - padding):
                         min(faceBox[2] + padding, frame.shape[1] - 1)]

        # Tạo một blob từ hình ảnh khuôn mặt để chuẩn bị cho việc nhận dạng giới tính và tuổi tác
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
        # Dự đoán giới tính của khuôn mặt
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')  # In giới tính lên console

            ageNet.setInput(blob)
        # Dự đoán số tuổi của khuôn mặt
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')    # In số tuổi lên console

        # Hiển thị văn bản bên trên hình ảnh với thông tin giới tính và tuổi tác
            cv2.putText(resultImg, f'{gender}, {age}', (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        

            if resultImg is None:
                continue

            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            # Chuyển đổi hình ảnh kết quả sang định dạng JPEG và gửi nó đi
            return (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['fileToUpload'].read()
        img = Image.open(io.BytesIO(f))
        img_ip = np.asarray(img, dtype="uint8")
        print(img_ip)
        return Response(gen_frames_photo(img_ip), mimetype='multipart/x-mixed-replace; boundary=frame')
        # return 'file uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)
