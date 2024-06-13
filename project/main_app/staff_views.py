import json
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import (HttpResponseRedirect, get_object_or_404,redirect, render)
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from .forms import *
from .models import *
from .models import CustomUser
from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from django.http import StreamingHttpResponse, Http404
from django.views.decorators import gzip
import io
import time

def staff_home(request):
    staff = get_object_or_404(Staff, admin=request.user)
    total_students = Student.objects.filter(course=staff.course).count()
    subjects = Subject.objects.filter(staff=staff)
    total_subject = subjects.count()
    attendance_list = Attendance.objects.filter(subject__in=subjects)
    total_attendance = attendance_list.count()
    attendance_list = []
    subject_list = []
    for subject in subjects:
        attendance_count = Attendance.objects.filter(subject=subject).count()
        subject_list.append(subject.name)
        attendance_list.append(attendance_count)
    context = {
        'page_title': 'Staff Panel - ' + str(staff.admin.last_name) + ' (' + str(staff.course) + ')',
        'total_students': total_students,
        'total_attendance': total_attendance,
        'total_subject': total_subject,
        'subject_list': subject_list,
        'attendance_list': attendance_list
    }
    return render(request, 'staff_template/home_content.html', context)


def staff_take_attendance(request):
    staff = get_object_or_404(Staff, admin=request.user)
    subjects = Subject.objects.filter(staff_id=staff)
    sessions = Session.objects.all()
    context = {
        'subjects': subjects,
        'sessions': sessions,
        'page_title': 'Take Attendance'
    }

    return render(request, 'staff_template/staff_take_attendance.html', context)
#######################################################################################################################

import cv2
import numpy as np
import pickle
import face_recognition
detected_name = set()

def recognize():
    with open("media/model/face_data.pkl", 'rb') as file:
        data = pickle.load(file)

    known_face_encodings = data["known_face_encodings"]
    known_face_names = data["known_face_names"]

    threshold = 0.30
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("media/model/haarcascade_frontalface_default.xml")

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_encodings = []
        face_locations = []
        face_names = []

        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))
            face_encodings.append(face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0])

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            if matches[best_match_index] and face_distances[best_match_index] >= threshold:
                name = known_face_names[best_match_index]
                detected_name.add(name)
            else:
                name = "Unknown"
            face_names.append(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (255, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            
            text_width, text_height = cv2.getTextSize(name, font, 0.5, 1)[0]

            text_x = left + (right - left - text_width) // 2
            text_y = bottom - 5

            cv2.putText(frame, name, (text_x, text_y), font, 0.5, (0, 255, 0), 1)
       
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
@gzip.gzip_page
def webcam_stream(request):
    try:
        return StreamingHttpResponse(recognize(), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print("Server Error:", e)
        raise Http404("Failed to open webcam.")
#################################################################################################


@csrf_exempt
def sav_attendance(request):
    time.sleep(10)
    student_data = request.POST.get('student_ids')
    date = request.POST.get('date')
    subject_id = request.POST.get('subject')
    session_id = request.POST.get('session')
    students = json.loads(student_data)
    global detected_name
    print(detected_name)
    status_dict = []

    try:
        session = get_object_or_404(Session, id=session_id)
        subject = get_object_or_404(Subject, id=subject_id)
        attendance = Attendance(session=session, subject=subject, date=date)
        attendance.save()
        for student_dict in students:
            student = get_object_or_404(Student, id=student_dict.get('id'))
            first_name = student.admin.first_name
            if first_name in detected_name:
                data = {
                    "id": student.id,
                    "status": 1
                    }
                status_dict.append(data)
            else:
                data = {
                    "id": student.id,
                    "status": 0
                    }
                status_dict.append(data)
        for student_dic in status_dict:
            student = get_object_or_404(Student, id=student_dic.get('id'))
            attendance_report = AttendanceReport(student=student, attendance=attendance, status=student_dic.get('status'))
            attendance_report.save()
    except Exception as e:
        return None
    return HttpResponse("OK")


################################################################################################################################


@csrf_exempt
def get_students(request):
    subject_id = request.POST.get('subject')
    session_id = request.POST.get('session')
    try:
        subject = get_object_or_404(Subject, id=subject_id)
        session = get_object_or_404(Session, id=session_id)
        students = Student.objects.filter(
            course_id=subject.course.id, session=session)
        student_data = []
        for student in students:
            data = {
                    "id": student.id,
                    "name": student.admin.first_name + " " + student.admin.last_name
                    }
            student_data.append(data)
            
        return JsonResponse(json.dumps(student_data), content_type='application/json', safe=False)
    except Exception as e:
        return e
    

@csrf_exempt
def save_attendance(request):
    student_data = request.POST.get('student_ids')
    date = request.POST.get('date')
    subject_id = request.POST.get('subject')
    session_id = request.POST.get('session')
    students = json.loads(student_data)
    try:
        session = get_object_or_404(Session, id=session_id)
        subject = get_object_or_404(Subject, id=subject_id)
        attendance = Attendance(session=session, subject=subject, date=date)
        attendance.save()

        for student_dict in students:
            student = get_object_or_404(Student, id=student_dict.get('id'))
            attendance_report = AttendanceReport(student=student, attendance=attendance, status=student_dict.get('status'))
            attendance_report.save()
    except Exception as e:
        return None
    return HttpResponse("OK")


def staff_update_attendance(request):
    staff = get_object_or_404(Staff, admin=request.user)
    subjects = Subject.objects.filter(staff_id=staff)
    sessions = Session.objects.all()
    context = {
        'subjects': subjects,
        'sessions': sessions,
        'page_title': 'Update Attendance'
    }
    return render(request, 'staff_template/staff_update_attendance.html', context)


@csrf_exempt
def get_student_attendance(request):
    attendance_date_id = request.POST.get('attendance_date_id')
    try:
        date = get_object_or_404(Attendance, id=attendance_date_id)
        attendance_data = AttendanceReport.objects.filter(attendance=date)
        student_data = []
        for attendance in attendance_data:
            data = {"id": attendance.student.admin.id,
                    "name": attendance.student.admin.last_name + " " + attendance.student.admin.first_name,
                    "status": attendance.status}
            student_data.append(data)
        return JsonResponse(json.dumps(student_data), content_type='application/json', safe=False)
    except Exception as e:
        return e


@csrf_exempt
def update_attendance(request):
    student_data = request.POST.get('student_ids')
    date = request.POST.get('date')
    students = json.loads(student_data)
    try:
        attendance = get_object_or_404(Attendance, id=date)

        for student_dict in students:
            student = get_object_or_404(
                Student, admin_id=student_dict.get('id'))
            attendance_report = get_object_or_404(AttendanceReport, student=student, attendance=attendance)
            attendance_report.status = student_dict.get('status')
            attendance_report.save()
    except Exception as e:
        return None
    return HttpResponse("OK")


@csrf_exempt
def staff_fcmtoken(request):
    token = request.POST.get('token')
    try:
        staff_user = get_object_or_404(CustomUser, id=request.user.id)
        staff_user.fcm_token = token
        staff_user.save()
        return HttpResponse("True")
    except Exception as e:
        return HttpResponse("False")


@csrf_exempt
def view_student_leave(request):
    if request.method != 'POST':
        allLeave = LeaveReportStudent.objects.all()
        context = {
            'allLeave': allLeave,
            'page_title': 'Leave Applications From Students'
        }
        return render(request, "staff_template/student_leave_view.html", context)
    else:
        id = request.POST.get('id')
        status = request.POST.get('status')
        if (status == '1'):
            status = 1
        else:
            status = -1
        try:
            leave = get_object_or_404(LeaveReportStudent, id=id)
            leave.status = status
            leave.save()
            return HttpResponse(True)
        except Exception as e:
            return False
        


########################################################################################################
